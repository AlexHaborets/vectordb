from __future__ import annotations


from pydantic import BaseModel

from src.common.metrics import MetricType


class CollectionBase(BaseModel):
    name: str
    dimension: int
    metric: MetricType

class CollectionCreate(CollectionBase):
    pass


class Collection(CollectionBase):
    id: int

    class Config:
        from_attributes = True

class IndexMetadata(BaseModel):
    key: str
    value: str