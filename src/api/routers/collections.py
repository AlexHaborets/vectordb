from typing import Annotated, List

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.api.exceptions import CollectionNotFoundError
from src.db.crud import VectorDBRepository
from src.schemas import Collection, CollectionCreate


collection_router = APIRouter(prefix="/collections", tags=["collections"])



@collection_router.post("/", response_model=Collection, status_code=201)
def add(
    request: Request,
    collection: CollectionCreate,
    db: Annotated[Session, Depends(get_db_session)],
) -> Collection:
    return VectorDBRepository(db).create_collection(collection)


@collection_router.get("/all", response_model=List[Collection])
def get_all(db: Annotated[Session, Depends(get_db_session)]) -> List[Collection]:
    return VectorDBRepository(db).get_all_collections()


@collection_router.get("/{collection_name}", response_model=Collection)
def get_collection(collection_name: str, db: Annotated[Session, Depends(get_db_session)]):
    collection = VectorDBRepository(db).get_collection_by_name(collection_name)
    if not collection:
        raise CollectionNotFoundError(collection_name)
    return collection