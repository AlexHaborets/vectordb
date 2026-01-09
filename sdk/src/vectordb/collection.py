from typing import Any, Dict, List, Optional, Sequence

from vectordb.models import SearchResult, Vector
from vectordb.transport import Transport
import numpy as np


class Collection:
    def __init__(self, name: str, transport: Transport) -> None:
        self.name = name
        self._transport = transport

    def upsert(
        self,
        ids: Sequence[str],
        vectors: np.ndarray | Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        batch_size: int = 64,
    ) -> bool:
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        if metadatas is None:
            metadatas = [None] * len(ids)

        if not len(ids) == len(vectors) == len(metadatas):
            raise ValueError(
                f"Length mismatch: ids={len(ids)}, vectors={len(vectors)}, metadatas={len(metadatas)}"
            )

        total = len(ids)
        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            batch = [
                {"id": _id, "vector": vec, "metadata": meta}
                for _id, vec, meta in zip(batch_ids, batch_vectors, batch_metas)
            ]

            self._transport.post(
                path=f"/collections/{self.name}/vectors", json={"vectors": batch}
            )

        return True

    def search(self, query: List[float] | np.ndarray, k: int = 5) -> List[SearchResult]:
        if isinstance(query, np.ndarray):
            query = query.tolist()
        payload = {"vector": query, "k": k}
        path = f"/collections/{self.name}/search"
        response = self._transport.post(path=path, json=payload)
        if not response:
            return []
        return [SearchResult.model_validate(item) for item in response]

    def get(self, vector_id: str) -> Vector:
        path = f"/collections/{self.name}/vectors/{vector_id}"
        response = self._transport.get(path=path)
        return Vector.model_validate(response)
