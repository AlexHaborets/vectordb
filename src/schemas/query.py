from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel, computed_field

from src.common import config
from src.schemas.vector import Vector

class Query(BaseModel):
    vector: List[float]

    @computed_field
    @property
    def numpy_vector(self) -> np.ndarray:
        return np.array(self.vector, dtype=config.NUMPY_DTYPE)
    
    class Config:
        arbitrary_types_allowed = True

class SearchResult(BaseModel):
    score: float
    vector: Vector