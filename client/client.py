from typing import List
from client import models
from client.collection import Collection
from client.errors import NotFoundError
from client.transport import Transport
from client.config import DEFAULT_DB_URL
from client.models import Metric


class Client:
    def __init__(self, url: str = DEFAULT_DB_URL) -> None:
        self._transport = Transport(base_url=url)

    def create_collection(self, name: str, dimension: int, metric: Metric) -> Collection:
        payload = models.Collection(
            name=name, 
            dimension=dimension, 
            metric=metric
        )
        self._transport.post("/collections/", json=payload.model_dump())
        return Collection(name, self._transport)

    def get_collection(self, name: str) -> Collection:
        self._transport.get(f"/collections/{name}")
        return Collection(name, self._transport)
    
    def get_or_create_collection(self, name: str, dimension: int, metric: Metric) -> Collection:
        collection: Collection
        try: 
            collection = self.get_collection(name)
        except NotFoundError:
            collection = self.create_collection(name, dimension, metric)
        return collection

    def list_collections(self) -> List[str]:
        data = self._transport.get("/collections/")
        return [item["name"] for item in data]

    def delete_collection(self, name: str) -> None:
        self._transport.delete(f"/collections/{name}")
