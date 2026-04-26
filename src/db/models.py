from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import (
    BLOB,
    JSON,
    Enum,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.common.config import NUMPY_DTYPE
from src.common.metrics import MetricType


class Base(DeclarativeBase):
    pass


class Vector(Base):
    __tablename__ = "vectors"

    __table_args__ = (
        UniqueConstraint(
            "collection_id", "external_id", name="uq_collection_external_id"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )

    vector_blob: Mapped[bytes] = mapped_column(BLOB, nullable=False)

    collection: Mapped["Collection"] = relationship(back_populates="vectors")

    vector_metadata: Mapped[Dict] = mapped_column(JSON, nullable=True)

    @property
    def numpy_vector(self) -> NDArray[NUMPY_DTYPE]:
        return np.frombuffer(self.vector_blob, dtype=NUMPY_DTYPE)

    @numpy_vector.setter
    def numpy_vector(self, value: NDArray[NUMPY_DTYPE]):
        self.vector_blob = value.astype(NUMPY_DTYPE).tobytes()

    @property
    def vector(self) -> List[float]:
        return self.numpy_vector.tolist()

    @vector.setter
    def vector(self, value: List[float]):
        self.numpy_vector = np.array(value)


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    metric: Mapped[MetricType] = mapped_column(
        Enum(MetricType), default=MetricType.COSINE, nullable=False
    )

    vectors: Mapped[List["Vector"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )

    index_metadata: Mapped[List["IndexMetadata"]] = relationship(
        "IndexMetadata", back_populates="collection", cascade="all, delete-orphan"
    )

    index_snapshot: Mapped[Optional["IndexSnapshot"]] = relationship(
        "IndexSnapshot",
        back_populates="collection",
        cascade="all, delete-orphan",
        uselist=False,
    )


class IndexMetadata(Base):
    # A helper key-value store
    __tablename__ = "index_metadata"

    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True
    )

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)

    collection: Mapped["Collection"] = relationship(
        "Collection", back_populates="index_metadata"
    )


class IndexSnapshot(Base):
    __tablename__ = "index_snapshots"

    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True
    )

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_point_id: Mapped[int] = mapped_column(Integer, nullable=True)
    payload: Mapped[bytes] = mapped_column(BLOB, nullable=False)

    collection: Mapped["Collection"] = relationship(
        "Collection", back_populates="index_snapshot"
    )
