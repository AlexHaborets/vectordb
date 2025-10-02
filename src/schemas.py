from pydantic import BaseModel
from typing import List


class MetadataBase(BaseModel):
    source_document: str
    content: str

class MetadataCreate(MetadataBase):
    pass

class Metadata(MetadataBase):
    vector_id: int

    class Config:
        from_attributes = True

class VectorBase(BaseModel):
    pass

class VectorCreate(VectorBase):
    vector: List[float]
    metadata: MetadataCreate

class Vector(VectorBase):
    id: int
    metadata: Metadata

    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    score: float
    vector: Vector
