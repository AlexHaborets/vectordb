from typing import List

from client.models import Query, SearchResult, Vector, UpsertBatch
from client.transport import Transport


class Collection:
    def __init__(self, name: str, transport: Transport) -> None:
        self.name = name
        self._transport = transport

    def upsert(self, vectors: List[Vector]) -> List[Vector]:
        payload = UpsertBatch(vectors=vectors)
        path = f"/collections/{self.name}/vectors"
        response = self._transport.post(path=path, json=payload.model_dump())
        return [Vector.model_validate(v) for v in response]
    
    def search(self, query: List[float], k: int = 5) -> List[SearchResult]:
        payload = Query(
            vector=query,
            k=k
        )
        path = f"/collections/{self.name}/search"
        response = self._transport.post(path=path, json=payload.model_dump())
        return [SearchResult.model_validate(item) for item in response]
    
    def get(self, vector_id: str) -> Vector:
        path = f"/collections/{self.name}/vectors/{vector_id}"
        response = self._transport.get(path=path)
        return Vector.model_validate(response)    