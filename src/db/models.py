from typing import List

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import (
    BLOB,
    Boolean,
    Column,
    Enum,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.common.config import NUMPY_DTYPE
from src.common.metrics import MetricType


class Base(DeclarativeBase):
    pass


graph_association_table = Table(
    "graph",
    Base.metadata,
    Column(
        "source_id",
        Integer,
        ForeignKey("vectors.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "neighbor_id",
        Integer,
        ForeignKey("vectors.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "collection_id",
        Integer,
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True, 
    )
)


class Vector(Base):
    __tablename__ = "vectors"

    __table_args__ = (
        UniqueConstraint('collection_id', 'external_id', name='uq_collection_external_id'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)

    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )

    vector_blob: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )

    collection: Mapped["Collection"] = relationship(back_populates="vectors")

    neighbors: Mapped[List["Vector"]] = relationship(
        "Vector",
        secondary=graph_association_table,
        primaryjoin=(id == graph_association_table.c.source_id),
        secondaryjoin=(id == graph_association_table.c.neighbor_id),
    )

    vector_metadata: Mapped["VectorMetadata"] = relationship(
        back_populates="vector", cascade="all, delete-orphan"
    )

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


class VectorMetadata(Base):
    __tablename__ = "vector_metadata"

    vector_id: Mapped[str] = mapped_column(
        ForeignKey("vectors.id", ondelete="CASCADE"), primary_key=True
    )

    source_document: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    vector: Mapped["Vector"] = relationship(back_populates="vector_metadata")


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    metric: Mapped[MetricType] = mapped_column(
        Enum(MetricType), 
        default=MetricType.COSINE,
        nullable=False
    )

    vectors: Mapped[List["Vector"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )
    
    index_metadata: Mapped[List["IndexMetadata"]] = relationship(
        "IndexMetadata",
        back_populates="collection",
        cascade="all, delete-orphan"
    )


class IndexMetadata(Base):
    # A helper key-value store
    __tablename__ = "index_metadata"

    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), 
        primary_key=True
    )

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)

    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="index_metadata"
    )