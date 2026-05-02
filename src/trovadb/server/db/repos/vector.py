from typing import Dict, List, Optional

from sqlalchemy import delete, func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

import trovadb.server.db.models as models
import trovadb.server.schemas as schemas
from trovadb.server.common.utils import vector_to_bytes


class VectorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _base_vector_query(self, collection_id: int) -> Query[models.Vector]:
        query = self.session.query(models.Vector).filter(
            models.Vector.collection_id == collection_id
        )

        return query

    def get_vector_by_external_id(
        self, collection_id: int, vector_id: str
    ) -> Optional[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .filter(models.Vector.external_id == vector_id)
            .one_or_none()
        )

    def get_vector_by_id(
        self, collection_id: int, vector_id: int
    ) -> Optional[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .filter(models.Vector.id == vector_id)
            .one_or_none()
        )

    def get_vectors_by_external_id(
        self, collection_id: int, vector_ids: List[str]
    ) -> List[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .filter(models.Vector.external_id.in_(vector_ids))
            .all()
        )

    def get_vectors_by_id(
        self, collection_id: int, vector_ids: List[int]
    ) -> List[models.Vector]:
        return (
            self._base_vector_query(collection_id)
            .filter(models.Vector.id.in_(vector_ids))
            .all()
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

    def upsert_vectors(
        self,
        collection_id: int,
        vectors: List[schemas.VectorCreate],
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
                    "vector_metadata": v.metadata if v.metadata else None,
                }
            )

        stmt = insert(models.Vector).values(vector_values)

        stmt = stmt.on_conflict_do_update(
            index_elements=["collection_id", "external_id"],
            set_={
                "vector_blob": stmt.excluded.vector_blob,
                "vector_metadata": stmt.excluded.vector_metadata,
            },
        )

        # Instead of refreshing, we are going to construct the list of models.Vector
        # by ourselves using the return sql stmt
        stmt = stmt.returning(models.Vector)

        vectors_in_db = self.session.execute(stmt).scalars().all()

        return list(vectors_in_db)

    def delete_vectors(
        self, collection_id: int, vector_ids: List[str]
    ) -> List[models.Vector]:
        if not vector_ids:
            return []

        stmt = (
            delete(models.Vector)
            .where(models.Vector.collection_id == collection_id)
            .where(models.Vector.external_id.in_(vector_ids))
            .returning(models.Vector)
        )
        deleted_vectors = self.session.execute(stmt).scalars().all()

        return list(deleted_vectors)
