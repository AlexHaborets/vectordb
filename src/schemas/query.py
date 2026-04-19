from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, computed_field, model_validator

from src.common import config
from src.schemas.vector import Vector


class Query(BaseModel):
    vector: List[float]
    k: int
    L_search: Optional[int] = Field(
        None,
        description="Search list size. Controls recall vs latency. Must be greater than or equal to k.",
    )

    @computed_field
    @property
    def numpy_vector(self) -> np.ndarray:
        return np.array(self.vector, dtype=config.NUMPY_DTYPE)

    @model_validator(mode="after")
    def validate_l_search(self) -> "Query":
        if self.L_search is None:
            return self

        if self.L_search < self.k:
            raise ValueError(
                f"L_search ({self.L_search}) cannot be less than k ({self.k})"
            )

        if self.L_search > config.MAX_L_SEARCH:
            raise ValueError("L_search cannot exceed 1000")

        return self

    class Config:
        arbitrary_types_allowed = True


class SearchResult(BaseModel):
    score: float
    vector: Vector
