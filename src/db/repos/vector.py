from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import func, select, update
from sqlalchemy.orm import Query, joinedload
from sqlalchemy.orm.session import Session

from src.common.utils import vector_to_bytes
import src.db.models as models
import src.schemas as schemas


class VectorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _base_vector_query(
        self, collection_id: int, include_metadata: bool
    ) -> Query[models.Vector]:
        query = self.session.query(models.Vector).filter(
            models.Vector.collection_id == collection_id,
            models.Vector.deleted == False,  # noqa: E712
        )
        if include_metadata:
            query = query.options(joinedload(models.Vector.vector_metadata))

        return query

    def get_vector_by_id(
        self, collection_id: int, vector_id: str, include_metadata: bool = True
    ) -> Optional[models.Vector]:
        return (
            self._base_vector_query(collection_id, include_metadata)
            .filter(models.Vector.external_id == vector_id)
            .one_or_none()
        )

    def get_all_vectors(
        self, collection_id: int, include_metadata: bool = True
    ) -> List[models.Vector]:
        return self._base_vector_query(collection_id, include_metadata).all()

    def get_random_sample(
        self, collection_id: int, size: int = 1000, include_metadata: bool = True
    ) -> List[models.Vector]:
        return (
            self._base_vector_query(collection_id, include_metadata)
            .order_by(func.random())
            .limit(size)
            .all()
        )

    def get_vectors_by_ids(
        self, collection_id: int, ids: List[int], include_metadata: bool = True
    ) -> List[models.Vector]:
        return (
            self._base_vector_query(collection_id, include_metadata)
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

        id_map = {v.external_id: v.id for v in vectors_in_db}

        metadata_data = [
            {
                "vector_id": id_map[v.id],
                "source": v.metadata.source,
                "content": v.metadata.content,
            }
            for v in vectors
            if v.id in id_map
        ]

        if metadata_data:
            stmt = insert(models.VectorMetadata).values(metadata_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["vector_id"],
                set_={"source": stmt.excluded.source, "content": stmt.excluded.content},
            )
            self.session.execute(stmt)

        self.session.commit()

        meta_map = {meta["vector_id"]: meta for meta in metadata_data}

        # "Refresh" vectors inplace instead of doing a query which is slow
        for v in vectors_in_db:
            if v.id in meta_map:
                metadata = meta_map[v.id]
                v.vector_metadata = models.VectorMetadata(
                    vector_id=v.id,
                    source=metadata["source"],
                    content=metadata["content"],
                )

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
        self.session.commit()
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

    def save_graph(self, collection_id: int, graph: Dict[int, List[int]]) -> None:
        self.session.execute(
            models.graph_association_table.delete().where(
                models.graph_association_table.c.collection_id == collection_id
            )
        )
        self.session.flush()

        if not graph:
            self.session.commit()
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

        self.session.commit()

    def get_unindexed_vector_ids(self, collection_id: int) -> List[int]:
        # Perform a left outer join to get the ids of vectors that are in collection
        # but not in the graph
        stmt = (
            select(models.Vector.id)
            .outerjoin(
                target=models.graph_association_table,
                onclause=(models.Vector.id == models.graph_association_table.c.source_id) &
                         (models.Vector.collection_id == models.graph_association_table.c.collection_id)
            )
            .where(
                models.Vector.collection_id == collection_id,
                models.Vector.deleted == False, # noqa: E712
                models.graph_association_table.c.source_id == None  # noqa: E711
            )
        )
        
        result = self.session.execute(stmt)
        return list(result.scalars().all())