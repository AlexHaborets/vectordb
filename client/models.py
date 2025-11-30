from typing import List
from pydantic import BaseModel
from client.types import Metric


class VectorMetadata(BaseModel):
    source_document: str
    content: str


class Vector(BaseModel):
    vector: List[float]
    vector_metadata: VectorMetadata


class VectorResponse(Vector):
    id: int


class SearchResult(BaseModel):
    score: float
    vector: VectorResponse


class Query(BaseModel):
    vector: List[float]
    k: int


class Collection(BaseModel):
    name: str
    dimension: int
    metric: Metric
