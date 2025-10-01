from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, ForeignKey, Table, INTEGER

class Base(DeclarativeBase):
    pass

graph_association_table = Table(
    'graph', Base.metadata,
    Column('source_id', INTEGER, 
           ForeignKey('vectors.id', ondelete='CASCADE'), primary_key=True),
    Column('neighbor_id', INTEGER, 
           ForeignKey('vectors.id', ondelete='CASCADE'), primary_key=True)
)
