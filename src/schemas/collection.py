from __future__ import annotations


from pydantic import BaseModel, Field

from src.common.config import MAX_COLLECTION_NAME_LENGTH, MAX_DIMENSIONS, MIN_DIMENSIONS
from src.common.metrics import MetricType


class CollectionBase(BaseModel):
    name: str = Field(
        min_length=3,
        max_length=MAX_COLLECTION_NAME_LENGTH,
        pattern=r"^[a-z0-9][a-z0-9_-]*$", 
        description="Must start with alphanumeric, allow only a-z, 0-9, _, -"
    )

    dimension: int = Field(
        ge=MIN_DIMENSIONS, 
        le=MAX_DIMENSIONS,
        description=f"Vector dimension ({MIN_DIMENSIONS}-{MAX_DIMENSIONS})"
    )

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
