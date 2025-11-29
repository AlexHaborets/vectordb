from typing import Annotated, List

from fastapi import APIRouter, Depends

from src.api.dependencies import get_indexer_manager, get_uow
from src.db import UnitOfWork
from src.engine import IndexerManager
from src.schemas import Vector, VectorCreate
from src.services import CollectionService, SearchService

vector_router = APIRouter(prefix="/collections/{collection_name}/vectors", tags=["vectors"])

collection_service = CollectionService()
search_service = SearchService()

@vector_router.post("/", response_model=Vector, status_code=201)
def add(
    collection_name: str,
    vector: VectorCreate,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> Vector:
    with uow:
        vector_db = collection_service.add_vector(collection_name, vector, uow)
        search_service.update(
            collection_name=collection_name,
            vector=vector_db,
            indexer_manager=indexer_manager,
            uow=uow
        )
        return vector_db


@vector_router.get("/", response_model=Vector)
def get_by_id(
    collection_name: str, vector_id: int, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> Vector:
    with uow:
        return collection_service.get_vector(collection_name, vector_id, uow)


@vector_router.get("/", response_model=List[Vector])
def get_all(
    collection_name: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> List[Vector]:
    with uow:
        return collection_service.get_all_vectors(collection_name, uow)


@vector_router.delete("/", response_model=Vector, status_code=201)
def delete_by_id(
    collection_name: str,
    vector_id: int,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
) -> Vector:
    with uow:
        return collection_service.delete_vector(collection_name, vector_id, uow)