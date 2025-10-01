from pydantic import BaseModel
from typing import List
from metadata import Metadata, MetadataCreate

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