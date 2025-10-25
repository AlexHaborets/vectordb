from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, field_validator

from src.config import NUMPY_DTYPE


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

    @field_validator("vector", mode="before")
    @classmethod
    def vector_to_list(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    class Config:
        from_attributes = True

class Query(BaseModel):
    vector: List[float]

class VectorLite(VectorBase):
    id: int
    numpy_vector: np.ndarray

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, VectorLite):
            return NotImplemented
        return self.id == other.id

    @classmethod
    def from_query(cls, query: Query) -> "VectorLite":
        numpy_array = np.array(query.vector, dtype=NUMPY_DTYPE)
        return cls(id=-1, numpy_vector=numpy_array)

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True


class SearchResult(BaseModel):
    vector: List[float]

    @classmethod
    def from_vector_lite(cls, v: VectorLite) -> "SearchResult":
        return cls(vector=v.numpy_vector.tolist())


class IndexMetadata(BaseModel):
    key: str
    value: str
