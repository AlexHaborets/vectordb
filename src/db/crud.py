from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.session import Session

import src.db.models as models
import src.schemas as schemas


class VectorDBRepository:
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
    
    def get_all_vectors_lite(self) -> List[schemas.VectorLite]:
        vectors = (
            self.session.query(models.Vector)
            .filter(
                models.Vector.deleted == False  # noqa: E712
            )
            .all()
        )
        return [schemas.VectorLite.model_validate(vec) for vec in vectors]
    
    def get_random_sample(self, size: int = 1000) -> List[schemas.VectorLite]:
        vectors = (
            self.session.query(models.Vector)
            .filter(models.Vector.deleted == False)  # noqa: E712
            .order_by(func.random())
            .limit(size)
            .all()
        )
        return [schemas.VectorLite.model_validate(vec) for vec in vectors]

    def get_vector_by_id(self, vector_id: int) -> Optional[schemas.Vector]:
        vector = self._get_vector_by_id(vector_id)
        if vector and not vector.deleted:
            return schemas.Vector.model_validate(vector)
        return None
    
    def get_vector_by_id_lite(self, vector_id: int) -> Optional[schemas.VectorLite]:
        vector = self._get_vector_by_id(vector_id)
        if vector and not vector.deleted:
            return schemas.VectorLite.model_validate(vector)
        return None
    
    def get_vectors_by_ids_lite(self, ids: List[int]) -> List[schemas.VectorLite]:
        vectors = (
            self.session.query(models.Vector)
            .filter(models.Vector.deleted == False and models.Vector.id.in_(ids))  # noqa: E712
            .all()
        )
        return [schemas.VectorLite.model_validate(vec) for vec in vectors]

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
        graph = {v.id: [neighbor.id for neighbor in v.neighbors]
                 for v in vectors}
        return graph

    def save_graph(self, graph: Dict[int, List[int]]) -> None:
        self.session.execute(models.graph_association_table.delete())
        self.session.flush()

        if not graph:
            self.session.commit()
            return
        
        all_vector_ids = set(graph.keys())
        for neighbors in graph.values():
            all_vector_ids.update(neighbors)
        
        vectors_in_db = self.session.query(models.Vector).filter(
            models.Vector.id.in_(list(all_vector_ids))
        ).all()

        vector_map = {v.id: v for v in vectors_in_db}

        for vector_id, neighbor_ids in graph.items():
            vector = vector_map.get(vector_id)
            if not vector:
                continue
            vector.neighbors = [vector_map[n_id] for n_id in neighbor_ids if n_id in vector_map]

        self.session.commit()

    def get_vector_ids(self) -> List[int]:
        vector_ids = self.session.scalars(
            select(models.Vector.id)
        ).all()

        return list(vector_ids)

    def add_index_metadata(self, key: str, value: str):
        self.session.add(models.IndexMetadata(
            key = key,
            value = value
        ))
        self.session.commit()

    def get_index_metadata(self, key: str) -> Optional[str]:
        metadata = self.session.get(models.IndexMetadata, key)
        return metadata.value if metadata else None