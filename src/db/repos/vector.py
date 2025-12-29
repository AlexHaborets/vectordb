from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Query, joinedload
from sqlalchemy.orm.session import Session

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
        new_ids = [v.id for v in vectors]

        existing_vectors = (
            self.session.query(models.Vector)
            .filter(
                models.Vector.collection_id == collection_id,
                models.Vector.external_id.in_(new_ids),
            )
            .all()
        )

        existing_map = {v.external_id: v for v in existing_vectors}

        results = []

        for v in vectors:
            existing_vec = existing_map.get(v.id)

            if existing_vec:
                existing_vec.vector = v.vector

                if existing_vec.deleted:
                    existing_vec.deleted = False

                if existing_vec.vector_metadata:
                    existing_vec.vector_metadata.source = v.metadata.source
                    existing_vec.vector_metadata.content = v.metadata.content
                else:
                    # If for some obscure and unknown reason vector metadata doesn't exist:
                    existing_vec.vector_metadata = models.VectorMetadata(
                        source=v.metadata.source, content=v.metadata.content
                    )

                results.append(existing_vec)

            else:
                metadata = models.VectorMetadata(
                    source=v.metadata.source,
                    content=v.metadata.content,
                )
                new_vector = models.Vector(
                    collection_id=collection_id,
                    external_id=v.id,
                    vector=v.vector,
                    vector_metadata=metadata,
                )
                self.session.add(new_vector)
                results.append(new_vector)
            
        self.session.commit()

        for v in results:
            self.session.refresh(v)

        return results

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
                    {"source_id": src, "neighbor_id": neighbor, "collection_id": collection_id}
                )

                if len(batch) >= BATCH_SIZE:
                    self.session.execute(models.graph_association_table.insert(), batch)
                    batch = []
        if batch:
            self.session.execute(models.graph_association_table.insert(), batch)

        self.session.commit()
