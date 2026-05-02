from typing import Optional

from sqlalchemy import Integer, String, cast, delete
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm.session import Session

import trovadb.server.db.models as models


class IndexRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def bump_version(self, collection_id: int) -> int:
        stmt = insert(models.IndexMetadata).values(
            collection_id=collection_id,
            key="index_version",
            value="1",
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=["collection_id", "key"],
            set_={"value": cast(cast(models.IndexMetadata.value, Integer) + 1, String)},
        ).returning(models.IndexMetadata.value)

        return int(self.session.execute(stmt).scalar_one())

    def get_version(self, collection_id: int) -> int:
        value = self.get_index_metadata(collection_id, "index_version")
        return int(value) if value is not None else 0

    def get_index_metadata(self, collection_id: int, key: str) -> Optional[str]:
        metadata = self.session.get(models.IndexMetadata, (collection_id, key))
        return metadata.value if metadata else None

    def set_index_metadata(self, collection_id: int, key: str, value: str) -> None:
        self.session.merge(
            models.IndexMetadata(collection_id=collection_id, key=key, value=value)
        )

    def delete_index_metadata(self, collection_id: int, key: str) -> None:
        stmt = (
            delete(models.IndexMetadata)
            .where(models.IndexMetadata.collection_id == collection_id)
            .where(models.IndexMetadata.key == key)
        )
        self.session.execute(stmt)

    def save_snapshot(
        self, collection_id: int, version: int, entry_point_id: int, payload: bytes
    ) -> None:
        self.session.merge(
            models.IndexSnapshot(
                collection_id=collection_id,
                version=version,
                entry_point_id=entry_point_id,
                payload=payload,
            )
        )

    def get_snapshot(self, collection_id: int) -> Optional[models.IndexSnapshot]:
        return self.session.get(models.IndexSnapshot, collection_id)

    def delete_snapshot(self, collection_id: int) -> None:
        stmt = delete(models.IndexSnapshot).where(
            models.IndexSnapshot.collection_id == collection_id
        )
        self.session.execute(stmt)
