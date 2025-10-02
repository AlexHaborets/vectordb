from pydantic import BaseModel
from typing import List


class VectorMetadataBase(BaseModel):
    source_document: str
    content: str

class VectorMetadataCreate(VectorMetadataBase):
    pass

class Metadata(VectorMetadataBase):
    vector_id: int

    class Config:
        from_attributes = True

class VectorBase(BaseModel):
    pass

class VectorCreate(VectorBase):
    vector: List[float]
    metadata: VectorMetadataCreate

class Vector(VectorBase):
    id: int
    metadata: Metadata

    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    score: float
    vector: Vector
