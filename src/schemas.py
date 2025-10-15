from pydantic import BaseModel, computed_field, field_validator
from typing import List
import numpy as np

class VectorMetadataBase(BaseModel):
    source_document: str
    content: str

class VectorMetadataCreate(VectorMetadataBase):
    pass

class VectorMetadata(VectorMetadataBase):
    vector_id: int

    class Config:
        from_attributes = True

class VectorBase(BaseModel):
    pass

class VectorCreate(VectorBase):
    vector: List[float]
    vector_metadata: VectorMetadataCreate

class Vector(VectorBase):
    id: int
    vector: List[float]
    vector_metadata: VectorMetadata

    @field_validator('vector', mode='before')
    @classmethod
    def vector_to_list(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    
    class Config:
        from_attributes = True

class VectorInDB(Vector):
    deleted: bool   
    neighbors: List[Vector]

class SearchResult(BaseModel):
    score: float
    vector: Vector
