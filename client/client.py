from typing import List
from sdk import models
from sdk.collection import Collection
from sdk.transport import Transport


class Client:
    def __init__(self, url: str = "http://localhost:8000") -> None:
        self._transport = Transport(base_url=url)

    def create_collection(self, name: str, dimension: int) -> Collection:
        payload = models.Collection(name=name, dimension=dimension)
        self._transport.post("/collections/", json=payload.model_dump())
        return Collection(name, self._transport)

    def get_collection(self, name: str) -> Collection:
        self._transport.get(f"/collections/{name}")
        return Collection(name, self._transport)

    def list_collections(self) -> List[str]:
        data = self._transport.get("/collections/")
        return [item["name"] for item in data]
