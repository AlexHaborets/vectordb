from pydantic import BaseModel

class MetadataBase(BaseModel):
    source_document: str
    content: str

class MetadataCreate(MetadataBase):
    pass

class Metadata(MetadataBase):
    vector_id: int

    class Config:
        from_attributes = True