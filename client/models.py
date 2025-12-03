from typing import List
from pydantic import BaseModel
from client.types import Metric


class VectorMetadata(BaseModel):
    source: str
    content: str


class Vector(BaseModel):
    id: str
    vector: List[float]
    metadata: VectorMetadata

class UpsertBatch(BaseModel):
    vectors: List[Vector]
    
class SearchResult(BaseModel):
    score: float
    vector: Vector


class Query(BaseModel):
    vector: List[float]
    k: int


class Collection(BaseModel):
    name: str
    dimension: int
    metric: Metric
