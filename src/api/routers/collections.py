from typing import Annotated, List

from fastapi import APIRouter, Depends

from src.api.dependencies import get_indexer_manager, get_uow
from src.db import UnitOfWork
from src.engine import IndexerManager
from src.schemas import Collection, CollectionCreate
from src.services.collection import CollectionService

collection_router = APIRouter(prefix="/collections", tags=["collections"])

collection_service = CollectionService()

@collection_router.post("/", response_model=Collection, status_code=201)
def add(
    collection: CollectionCreate,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
) -> Collection:
    with uow:
        return collection_service.create_collection(collection, uow)


@collection_router.get("/", response_model=List[Collection])
def get_all(uow: Annotated[UnitOfWork, Depends(get_uow)]) -> List[Collection]:
    # TODO: Add pagination
    with uow:
        return collection_service.get_all_collections(uow)


@collection_router.get("/{collection_name}", response_model=Collection)
def get_collection(
    collection_name: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> Collection:
    with uow:
        return collection_service.get_collection_by_name(collection_name, uow)

@collection_router.delete("/{collection_name}", status_code=200)
def delete_collection(
    collection_name: str, 
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> None: 
    with uow:
        collection_service.delete_collection(collection_name, indexer_manager, uow)