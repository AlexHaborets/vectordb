from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm.session import Session

import src.db.models as models
import src.schemas as schemas

class VectorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_vector_by_id(self, collection_id: int, vector_id: int) -> Optional[models.Vector]:
        return (
            self.session.query(models.Vector)
            .filter(
                models.Vector.id == vector_id,
                models.Vector.collection_id == collection_id,
                models.Vector.deleted == False,  # noqa: E712
            )
            .one_or_none()
        )

    def get_all_vectors(self, collection_id: int) -> List[models.Vector]:
        return (
            self.session.query(models.Vector)
            .filter(
                models.Vector.collection_id == collection_id,
                models.Vector.deleted == False  # noqa: E712
            )
            .order_by(models.Vector.id)
            .all()
        )
    
    def get_random_sample(self, collection_id: int, size: int = 1000) -> List[models.Vector]:
        return (
            self.session.query(models.Vector)
            .filter(
                models.Vector.collection_id == collection_id,
                models.Vector.deleted == False  # noqa: E712
            )  
            .order_by(func.random())
            .limit(size)
            .all()
        )
    
    def get_vectors_by_ids(self, collection_id: int, ids: List[int]) -> List[models.Vector]:
        return (
            self.session.query(models.Vector)
            .filter(
                models.Vector.collection_id == collection_id,
                models.Vector.deleted == False, # noqa: E712
                models.Vector.id.in_(ids) 
            ) 
            .all()
        )
    
    def add_vector(self, collection_id: int, vector: schemas.VectorCreate) -> models.Vector:
        new_vector = models.Vector(
            vector_metadata=models.VectorMetadata(
                source_document=vector.vector_metadata.source_document,
                content=vector.vector_metadata.content,
            ),
            collection_id=collection_id,
            vector=vector.vector,
        )
        self.session.add(new_vector)
        self.session.commit()
        self.session.refresh(new_vector)
        return new_vector   

    def mark_vector_deleted(self,  collection_id: int, vector_id: int) -> Optional[models.Vector]:
        vector = self.get_vector_by_id(collection_id, vector_id)
        if vector and not vector.deleted:
            vector.deleted = True
            self.session.commit()
            self.session.refresh(vector)
        return vector

    def get_graph(self, collection_id: int) -> Dict[int, List[int]]:
        edges = self.session.execute(
            select(
                models.graph_association_table.c.source_id,
                models.graph_association_table.c.neighbor_id
            ).where(
                models.graph_association_table.c.collection_id == collection_id 
            )
        ).all()

        graph = defaultdict(list)
        for source, neighbor in edges:
            graph[source].append(neighbor)
        return graph

    def save_graph(self, collection_id: int, graph: Dict[int, List[int]]) -> None:
        self.session.execute(
            models.graph_association_table.delete()
            .where(
                models.graph_association_table.c.collection_id == collection_id
            )
        )
        self.session.flush()

        if not graph:
            self.session.commit()
            return
        
        edges = [
            {
                "source_id": src, 
                "neighbor_id": nbr, 
                "collection_id": collection_id
            }
            for src, neighbors in graph.items()
            for nbr in neighbors
        ]

        if not edges:
            return 
        
        self.session.execute(
            models.graph_association_table.insert()
            .values(edges) 
        )

        self.session.commit()

    def get_vector_ids(self) -> List[int]:
        vector_ids = self.session.scalars(
            select(models.Vector.id)
            .filter(models.Vector.deleted == False)  # noqa: E712
        ).all()

        return list(vector_ids)