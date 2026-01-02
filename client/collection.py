from typing import Any, Dict, List

from client.models import SearchResult, Vector
from client.transport import Transport
import numpy as np

class Collection:
    def __init__(self, name: str, transport: Transport) -> None:
        self.name = name
        self._transport = transport

    def upsert(
        self, 
        ids: List[str],
        vectors: np.ndarray | List[List[float]],
        metadatas: List[Dict[str, Any] | None] | None
        ) -> bool:
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        if metadatas is None:
            metadatas = [None] * len(ids) # type: ignore

        if not len(ids) == len(vectors) == len(metadatas): # type: ignore
            raise ValueError("Length mismatch")

        batch = [
            {
                "id": _id,
                "vector": vector,
                "metadata": metadata
            }
            for _id, vector, metadata in zip(ids, vectors, metadatas) # type: ignore
        ]

        payload = {"vectors": batch}
        path = f"/collections/{self.name}/vectors"
        self._transport.post(path=path, json=payload)
        return True
    
    def search(self, query: List[float], k: int = 5) -> List[SearchResult]:
        payload = {
            "vector": query,
            "k": k
        }
        path = f"/collections/{self.name}/search"
        response = self._transport.post(path=path, json=payload)
        return [SearchResult.model_validate(item) for item in response]
    
    def get(self, vector_id: str) -> Vector:
        path = f"/collections/{self.name}/vectors/{vector_id}"
        response = self._transport.get(path=path)
        return Vector.model_validate(response)    