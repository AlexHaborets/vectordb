from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import except_, func, select, update
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from src.common.utils import vector_to_bytes
import src.db.models as models
import src.schemas as schemas


class VectorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _base_vector_query(self, collection_id: int) -> Query[models.Vector]:
        query = self.session.query(models.Vector).filter(
            models.Vector.collection_id == collection_id,
            models.Vector.deleted == False,  # noqa: E712
        )

        return query

    def get_vector_by_id(
        self, collection_id: int, vector_id: str
    ) -> Optional[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .filter(models.Vector.external_id == vector_id)
            .one_or_none()
        )

    def get_all_vectors(self, collection_id: int) -> List[models.Vector]:
        return self._base_vector_query(collection_id).all()

    def get_random_sample(
        self, collection_id: int, size: int = 1000
    ) -> List[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .order_by(func.random())
            .limit(size)
            .all()
        )

    def get_vectors_by_ids(
        self, collection_id: int, ids: List[int]
    ) -> List[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .filter(models.Vector.id.in_(ids))
            .all()
        )

    def upsert_vectors(
        self, collection_id: int, vectors: List[schemas.VectorCreate]
    ) -> List[models.Vector]:
        if not vectors:
            return []

        vector_values: List[Dict] = []

        for v in vectors:
            vector_values.append(
                {
                    "collection_id": collection_id,
                    "external_id": v.id,
                    "vector_blob": vector_to_bytes(v.vector),
                    "deleted": False,
                    "vector_metadata": v.metadata if v.metadata else None,
                }
            )

        stmt = insert(models.Vector).values(vector_values)

        stmt = stmt.on_conflict_do_update(
            index_elements=["collection_id", "external_id"],
            set_={"vector_blob": stmt.excluded.vector_blob, "deleted": False},
        )

        # Instead of refreshing, we are going to construct the list of models.Vector
        # by ourselves using the return sql stmt
        stmt = stmt.returning(models.Vector)

        vectors_in_db = self.session.execute(stmt).scalars().all()
        
        return list(vectors_in_db)

    def mark_vector_deleted(self, collection_id: int, vector_id: str) -> bool:
        result = self.session.execute(
            update(models.Vector)
            .where(
                models.Vector.collection_id == collection_id,
                models.Vector.external_id == vector_id,
                models.Vector.deleted == False,  # noqa: E712
            )
            .values(deleted=True)
        )
        return result.rowcount > 0

    def get_graph(self, collection_id: int) -> Dict[int, List[int]]:
        edges = self.session.execute(
            select(
                models.graph_association_table.c.source_id,
                models.graph_association_table.c.neighbor_id,
            ).where(models.graph_association_table.c.collection_id == collection_id)
        ).all()

        graph = defaultdict(list)
        for source, neighbor in edges:
            graph[source].append(neighbor)
        return graph
    
    def update_graph(self, collection_id: int, subgraph: Dict[int, List[int]]):
        ids = subgraph.keys()
        if not ids:
            return
        
        self.session.execute(
            models.graph_association_table.delete().where(
                (models.graph_association_table.c.collection_id == collection_id) & 
                (models.graph_association_table.c.source_id.in_(ids)) 
            )
        )
        
        data = []
        for src, neighbors in subgraph.items():
            for neighbor in neighbors:
                data.append(
                    {
                        "source_id": src,
                        "neighbor_id": neighbor,
                        "collection_id": collection_id,
                    }
                )

        if data:
            self.session.execute(models.graph_association_table.insert(), data)

    def save_graph(
        self,
        collection_id: int,
        graph: Dict[int, List[int]],
    ) -> None:
        self.session.execute(
            models.graph_association_table.delete().where(
                models.graph_association_table.c.collection_id == collection_id
            )
        )

        if not graph:
            return

        BATCH_SIZE = 4096

        batch = []

        for src, neighbors in graph.items():
            for neighbor in neighbors:
                batch.append(
                    {
                        "source_id": src,
                        "neighbor_id": neighbor,
                        "collection_id": collection_id,
                    }
                )

                if len(batch) >= BATCH_SIZE:
                    self.session.execute(models.graph_association_table.insert(), batch)
                    batch = []

        if batch:
            self.session.execute(models.graph_association_table.insert(), batch)

    def get_unindexed_vector_ids(self, collection_id: int) -> List[int]:
        all_vectors_stmt = select(models.Vector.id).where(
            models.Vector.collection_id == collection_id,
            models.Vector.deleted == False,  # noqa: E712
        )

        indexed_vectors_stmt = select(models.graph_association_table.c.source_id).where(
            models.graph_association_table.c.collection_id == collection_id
        )

        stmt = except_(all_vectors_stmt, indexed_vectors_stmt)

        result = self.session.execute(stmt)
        return list(result.scalars().all())
