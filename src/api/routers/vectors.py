from typing import Annotated, List

from fastapi import APIRouter, Depends

from src.api.dependencies import get_indexer_manager, get_uow
from src.db import UnitOfWork
from src.engine import IndexerManager
from src.schemas import UpsertBatch, Vector
from src.services import CollectionService, IndexService

vector_router = APIRouter(
    prefix="/collections/{collection_name}/vectors", tags=["vectors"]
)

collection_service = CollectionService()
index_service = IndexService()


@vector_router.post("", response_model=List[Vector], status_code=201)
def upsert(
    collection_name: str,
    batch: UpsertBatch,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> List[Vector]:
    with uow:
        vectors_in_db = collection_service.upsert_vectors(
            collection_name, batch.vectors, uow
        )
        index_service.update(
            collection_name=collection_name,
            vectors=vectors_in_db,
            indexer_manager=indexer_manager,
            uow=uow,
        )
        return vectors_in_db


@vector_router.get("/{vector_id}", response_model=Vector)
def get_by_id(
    collection_name: str, vector_id: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> Vector:
    with uow:
        return collection_service.get_vector(collection_name, vector_id, uow)


@vector_router.get("", response_model=List[Vector])
def get_all(
    collection_name: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> List[Vector]:
    with uow:
        return collection_service.get_all_vectors(collection_name, uow)

# TODO: Implement deletion

# @vector_router.delete("", response_model=Vector, status_code=201)
# def delete_by_id(
#     collection_name: str,
#     vector_id: str,
#     uow: Annotated[UnitOfWork, Depends(get_uow)],
# ) -> None:
#     with uow:
#         collection_service.delete_vector(collection_name, vector_id, uow)
