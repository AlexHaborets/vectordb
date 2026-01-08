from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from src.common.config import (
    BATCH_SIZE_LIMIT,
    NUMPY_DTYPE,
    MAX_ID_LENGTH,
    MAX_META_SIZE,
)


class VectorBase(BaseModel):
    pass


class VectorCreate(VectorBase):
    id: str = Field(
        min_length=1,
        max_length=MAX_ID_LENGTH,
        pattern=r"^[a-zA-Z0-9_\-]+$",  # URL-safe
        description="Unique identifier for the vector",
    )

    vector: List[float] = Field(min_length=1)

    metadata: Dict[str, Any] | None = Field(default=None, max_length=MAX_META_SIZE)


class Vector(VectorBase):
    id: str = Field(validation_alias="external_id")
    internal_id: int = Field(validation_alias="id", exclude=True)

    vector: List[float]
    metadata: Dict[str, Any] | None = Field(
        validation_alias="vector_metadata", default=None
    )

    @field_validator("vector", mode="before")
    @classmethod
    def vector_to_list(cls, v: Any) -> List[float]:
        if isinstance(v, np.ndarray):
            return v.tolist()

        if not all(math.isfinite(x) for x in v):
            raise ValueError("Vector contains NaN or Inf values")

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
        return cls(id=v.internal_id, vector=v.numpy_vector)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True


class UpsertBatch(BaseModel):
    vectors: List[VectorCreate] = Field(min_length=1, max_length=BATCH_SIZE_LIMIT)

    @model_validator(mode="after")
    def check_batch_integrity(self) -> "UpsertBatch":
        ids = [v.id for v in self.vectors]
        if len(ids) != len(set(ids)):
            raise ValueError("Batch contains duplicate ids")
        return self
