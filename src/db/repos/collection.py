from typing import List, Optional

from sqlalchemy import delete, text
from sqlalchemy.orm.session import Session

import src.db.models as models
import src.schemas as schemas


class CollectionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_all_collections(self) -> List[models.Collection]:
        return self.session.query(models.Collection).all()

    def get_collection_by_name(
        self, collection_name: str
    ) -> Optional[models.Collection]:
        return (
            self.session.query(models.Collection)
            .where(models.Collection.name == collection_name)
            .one_or_none()
        )

    def create_collection(
        self, collection: schemas.CollectionCreate
    ) -> Optional[models.Collection]:
        if self.get_collection_by_name(collection.name):
            return None
        new_collection = models.Collection(
            name=collection.name,
            dimension=collection.dimension,
            metric=collection.metric,
        )
        self.session.add(new_collection)
        self.session.flush()
        self.session.refresh(new_collection)
        return new_collection

    def delete_collection(self, collection_id: int) -> None:
        self.session.execute(text("PRAGMA foreign_keys = OFF"))

        self.session.execute(
            delete(models.graph_association_table).where(
                models.graph_association_table.c.collection_id == collection_id
            )
        )

        self.session.execute(
            delete(models.Vector).where(models.Vector.collection_id == collection_id)
        )

        self.session.execute(
            delete(models.IndexMetadata).where(
                models.IndexMetadata.collection_id == collection_id
            )
        )

        self.session.execute(
            delete(models.Collection).where(models.Collection.id == collection_id)
        )

    def set_index_metadata(self, collection_id: int, key: str, value: str) -> None:
        self.session.merge(
            models.IndexMetadata(collection_id=collection_id, key=key, value=value)
        )

    def get_index_metadata(self, collection_id: int, key: str) -> Optional[str]:
        metadata = self.session.get(models.IndexMetadata, (collection_id, key))
        return metadata.value if metadata else None
