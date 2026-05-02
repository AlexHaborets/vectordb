from typing import Annotated, List

from fastapi import APIRouter, Depends, status

from trovadb.server.api.dependencies import get_indexer_manager, get_uow
from trovadb.server.db import UnitOfWork
from trovadb.server.engine import IndexerManager
from trovadb.server.schemas import DeleteBatch, UpsertBatch, Vector
from trovadb.server.schemas.vector import GetBatch
from trovadb.server.services import CollectionService

vector_router = APIRouter(
    prefix="/collections/{collection_name}/vectors", tags=["vectors"]
)

collection_service = CollectionService()


@vector_router.post(
    "", response_model=List[Vector], status_code=status.HTTP_201_CREATED
)
def upsert_batch(
    collection_name: str,
    batch: UpsertBatch,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> List[Vector]:
    with uow:
        return collection_service.upsert_vectors(
            collection_name, batch.vectors, indexer_manager, uow
        )


@vector_router.get("/{vector_id}", response_model=Vector)
def get(
    collection_name: str, vector_id: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> Vector:
    with uow:
        return collection_service.get_vector_by_external_id(
            collection_name, vector_id, uow
        )


@vector_router.post("/batch-get", response_model=List[Vector])
def get_batch(
    collection_name: str, batch: GetBatch, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> List[Vector]:
    with uow:
        return collection_service.get_vectors_by_external_id(
            collection_name, batch.ids, uow
        )


@vector_router.get("", response_model=List[Vector])
def get_all(
    collection_name: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> List[Vector]:
    with uow:
        return collection_service.get_all_vectors(collection_name, uow)


@vector_router.delete("/{vector_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(
    collection_name: str,
    vector_id: str,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> None:
    with uow:
        collection_service.delete_vectors(
            collection_name=collection_name,
            vector_ids=[vector_id],
            indexer_manager=indexer_manager,
            uow=uow,
        )


@vector_router.post("/delete", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch(
    collection_name: str,
    batch: DeleteBatch,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> None:
    with uow:
        collection_service.delete_vectors(
            collection_name=collection_name,
            vector_ids=batch.ids,
            indexer_manager=indexer_manager,
            uow=uow,
        )
