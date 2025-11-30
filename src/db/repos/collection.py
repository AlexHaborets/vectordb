from typing import List, Optional

from sqlalchemy.orm.session import Session

import src.db.models as models
import src.schemas as schemas

class CollectionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_all_collections(self) -> List[models.Collection]:
        return (
            self.session.query(models.Collection)
            .all()
        )

    def get_collection_by_name(self, collection_name: str) -> Optional[models.Collection]:
        return (
            self.session.query(models.Collection)
            .where(models.Collection.name == collection_name)
            .one_or_none()
        )
    
    
    def create_collection(self, collection: schemas.CollectionCreate) -> Optional[models.Collection]:
        if self.get_collection_by_name(collection.name):
            return None
        new_collection = models.Collection(
            name = collection.name,
            dimension = collection.dimension
        )
        self.session.add(new_collection)
        self.session.commit()
        self.session.refresh(new_collection)
        return new_collection
    
    def delete_collection(self, collection_name: str) -> bool:
        collection = self.get_collection_by_name(collection_name)
        if not collection:
            return False
        self.session.delete(collection)
        self.session.commit()
        return True

    
    def set_index_metadata(self, collection_id: int, key: str, value: str) -> None:
        self.session.merge(models.IndexMetadata(
            collection_id = collection_id,
            key = key,
            value = value
        ))
        self.session.commit()

    def get_index_metadata(self, collection_id: int, key: str) -> Optional[str]:
        metadata = self.session.get(models.IndexMetadata, (collection_id, key))
        return metadata.value if metadata else None