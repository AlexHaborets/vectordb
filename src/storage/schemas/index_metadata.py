from sqlalchemy import Column, String
from base import Base

class IndexMetadata(Base):
    # A helper key value store
    __tablename__ = "index_metadata"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)

    