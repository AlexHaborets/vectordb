import numpy as np
from sqlalchemy import BLOB, Column, ForeignKey, Integer, String, Table, Text
from sqlalchemy.orm import DeclarativeBase, relationship


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
)


class Vector(Base):
    __tablename__ = "vectors"
    id = Column(Integer, primary_key=True)
    vector = Column(BLOB, nullable=False)

    neighbors = relationship(
        "Vector",
        secondary=graph_association_table == graph_association_table.c.source_id,
        primaryjoin=id == graph_association_table.c.neighbor_id,
        backref="neighbor_of",
    )

    metadata = relationship(
        "Metadata", uselist=False, back_populates="vector", cascade="all, delete-orphan"
    )

    @property
    def numpy_vector(self):
        return np.frombuffer(self.vector_data, dtype=np.float32)

    @numpy_vector.setter
    def numpy_vector(self, value: np.ndarray):
        self.vector_data = value.astype(np.float32).tobytes()


class Metadata(Base):
    __tablename__ = "metadata"
    vector_id = Column(
        Integer, ForeignKey("vectors.id", ondelete="CASCADE"), primary_key=True
    )
    vector = Column(BLOB, nullable=False)

    source_document = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    vector = relationship("Vector", back_populates="metadata")


class IndexMetadata(Base):
    # A helper key value store
    __tablename__ = "index_metadata"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
