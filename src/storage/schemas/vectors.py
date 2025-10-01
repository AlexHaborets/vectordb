from sqlalchemy import Column, INTEGER, BLOB
from base import Base, graph_association_table
from sqlalchemy.orm import relationship
import numpy as np

class Vectors(Base):
    __tablename__ = "vectors"
    id = Column(INTEGER, primary_key=True)
    vector = Column(BLOB, nullable=False)

    neighbors = relationship(
        "Vector",
        secondary=graph_association_table == graph_association_table.c.source_id,
        primaryjoin=id == graph_association_table.c.neighbor_id,
        backref="neighbor_of"
    )

    metadata = relationship(
        "Metadata",
        uselist=False,
        back_populates="vector",
        cascade="all, delete-orphan"
    )

    @property
    def numpy_vector(self):
        return np.frombuffer(self.vector_data, dtype=np.float32)

    @numpy_vector.setter
    def numpy_vector(self, value: np.ndarray):
        self.vector_data = value.astype(np.float32).tobytes()