from __future__ import annotations

from typing import Any, List

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.common.config import BATCH_SIZE_LIMIT, NUMPY_DTYPE


class VectorMetadataBase(BaseModel):
    source: str
    content: str


class VectorMetadataCreate(VectorMetadataBase):
    pass


class VectorMetadata(VectorMetadataBase):
    class Config:
        from_attributes = True


class VectorBase(BaseModel):
    pass


class VectorCreate(VectorBase):
    id: str
    vector: List[float]
    metadata: VectorMetadataCreate


class Vector(VectorBase):
    id: str = Field(validation_alias="external_id")
    internal_id: int = Field(validation_alias="id", exclude=True)

    vector: List[float]
    metadata: VectorMetadata = Field(validation_alias="vector_metadata")

    @field_validator("vector", mode="before")
    @classmethod
    def vector_to_list(cls, v: Any) -> List[float]:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @property
    def numpy_vector(self) -> np.ndarray:
        return np.array(self.vector, dtype=NUMPY_DTYPE)

    class Config:
        from_attributes = True
        populate_by_name = True

class VectorData(VectorBase):
    id: int
    vector: np.ndarray = Field(validation_alias="numpy_vector")

    @classmethod
    def from_vector(cls, v) -> "VectorData":
        if not isinstance(v, Vector):
            raise ValueError(f"Expected Vector object. Got {type(v)}")
        return cls(
            id = v.internal_id,
            vector=v.numpy_vector
        )


    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True


class UpsertBatch(BaseModel):
    vectors: List[VectorCreate] = Field(min_length=1, max_length=BATCH_SIZE_LIMIT)
