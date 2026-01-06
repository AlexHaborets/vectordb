from typing import Any, Dict, List
from pydantic import BaseModel
from vectordb.types import Metric

class Vector(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    
class SearchResult(BaseModel):
    score: float
    vector: Vector

class Collection(BaseModel):
    name: str
    dimension: int
    metric: Metric
