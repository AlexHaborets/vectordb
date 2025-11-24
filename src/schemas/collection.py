from __future__ import annotations


from pydantic import BaseModel


class CollectionCreate(BaseModel):
    name: str
    dimension: int


class Collection(BaseModel):
    id: int
    name: str
    dimension: int

    class Config:
        from_attributes = True

class IndexMetadata(BaseModel):
    key: str
    value: str