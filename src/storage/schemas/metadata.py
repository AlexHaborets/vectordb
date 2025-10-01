from base import Base
from sqlalchemy import BLOB, INTEGER, Column, ForeignKey, String, Text
from sqlalchemy.orm import relationship


class Metadata(Base):
    __tablename__ = "metadata"
    vector_id = Column(
        INTEGER, ForeignKey("vectors.id", ondelete="CASCADE"), primary_key=True
    )
    vector = Column(BLOB, nullable=False)

    source_document = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    vector = relationship("Vector", back_populates="metadata")
