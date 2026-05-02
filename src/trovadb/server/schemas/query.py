from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, computed_field, model_validator

from trovadb.server.common import config
from trovadb.server.schemas.vector import Vector


class Query(BaseModel):
    vector: List[float]
    k: int = Field(..., description="The final number of vectors to return.")
    L_search: Optional[int] = Field(
        None,
        description="Search list size. Must be greater than or equal to k.",
        le=config.MAX_L_SEARCH,
    )

    # MMR params
    mmr_n: Optional[int] = Field(
        None,
        description="Number of candidates to fetch before MMR reranking. Must be >= k.",
    )
    mmr_lambda: Optional[float] = Field(
        None,
        description="MMR balance. 1.0 = relevance only, 0.0 = diversity only. Usually 0.5 - 0.7.",
        ge=0.0,
        le=1.0,
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
                f"L_search ({self.L_search}) cannot be less than k ({self.k})."
            )

        return self

    @model_validator(mode="after")
    def validate_mmr_n(self) -> "Query":
        if self.mmr_n is None:
            return self

        if self.mmr_n < self.k:
            raise ValueError(f"mmr_n ({self.mmr_n}) cannot be less than k ({self.k}).")

        return self

    class Config:
        arbitrary_types_allowed = True


class SearchResult(BaseModel):
    score: float
    vector: Vector
