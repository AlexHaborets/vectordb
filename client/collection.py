from typing import List

from client.models import Query, SearchResult, Vector, VectorMetadata, VectorResponse
from client.transport import Transport


class Collection:
    def __init__(self, name: str, transport: Transport) -> None:
        self.name = name
        self._transport = transport

    def add(self, vector: List[float], source: str, content: str) -> VectorResponse:
        payload = Vector(
            vector=vector, 
            vector_metadata=VectorMetadata(source_document=source, content=content)
        )
                
        path = f"/collections/{self.name}/vectors/"
        response = self._transport.post(path=path, json=payload.model_dump())
        return VectorResponse.model_validate(response)
    
    def search(self, query: List[float], k: int = 5) -> List[SearchResult]:
        payload = Query(
            vector=query,
            k=k
        )
        path = f"/collections/{self.name}/search/"
        response = self._transport.post(path=path, json=payload.model_dump())
        return [SearchResult.model_validate(item) for item in response]
    
    def get(self, vector_id: int) -> VectorResponse:
        path = f"/collections/{self.name}/vectors/?vector_id={vector_id}"
        response = self._transport.post(path=path)
        return VectorResponse.model_validate(response)    