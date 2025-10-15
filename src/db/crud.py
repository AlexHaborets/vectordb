from typing import Dict, List, Optional

from sqlalchemy.orm import selectinload
from sqlalchemy.orm.session import Session

import src.db.models as models
import src.schemas as schemas


class CRUDRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _get_vector_by_id(self, vector_id: int) -> Optional[models.Vector]:
        return self.session.get(models.Vector, vector_id)

    def get_all_vectors(self) -> List[schemas.Vector]:
        vectors = (
            self.session.query(models.Vector)
            .filter(
                models.Vector.deleted == False  # noqa: E712
            )
            .all()
        )
        return [schemas.Vector.model_validate(vec) for vec in vectors]

    def get_vector_by_id(self, vector_id: int) -> Optional[schemas.Vector]:
        vector = self._get_vector_by_id(vector_id)
        if vector and not vector.deleted:
            return schemas.Vector.model_validate(vector)
        return None

    def add_vector(self, vector: schemas.VectorCreate) -> schemas.Vector:
        new_vector = models.Vector(
            vector_metadata=models.VectorMetadata(
                source_document=vector.vector_metadata.source_document,
                content=vector.vector_metadata.content,
            ),
            vector=vector.vector,
        )
        self.session.add(new_vector)
        self.session.commit()
        self.session.refresh(new_vector, attribute_names=["vector_metadata"])
        return schemas.Vector.model_validate(new_vector)

    def mark_vector_deleted(self, vector_id: int) -> Optional[schemas.Vector]:
        vector = self._get_vector_by_id(vector_id)
        if vector and not vector.deleted:
            vector.deleted = True
            self.session.commit()
            self.session.refresh(vector)
            return schemas.Vector.model_validate(vector)
        return None

    def get_graph(self) -> Dict[int, List[int]]:
        vectors = (
            self.session.query(models.Vector)
            .options(selectinload(models.Vector.neighbors))
            .filter(models.Vector.deleted == False)  # noqa: E712
            .all()
        )
        graph = {v.id: [neighbor.id for neighbor in v.neighbors] for v in vectors}
        return graph
