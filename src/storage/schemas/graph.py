from sqlalchemy import Column, INTEGER
from base import Base


class Graph(Base):
    __tablename__ = "graph"
    # Both keys make up a composite primary key
    source_id = Column(INTEGER, primary_key=True)
    neighbor_id = Column(INTEGER, primary_key=True)
