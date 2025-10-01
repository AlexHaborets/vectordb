from pydantic import BaseModel
from vector import Vector

class SearchResult(BaseModel):
    score: float
    vector: Vector
