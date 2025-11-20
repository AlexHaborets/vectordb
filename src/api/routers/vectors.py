from typing import Annotated, List

from fastapi import APIRouter, Depends

from src.api.dependencies import get_indexer, get_uow
from src.common.exceptions import VectorNotFoundError
from src.engine import VamanaIndexer
from src.db import UnitOfWork
from src.schemas import Vector, VectorCreate, VectorLite
from src.services import CollectionService

vector_router = APIRouter(prefix="/{collection_name}/vectors", tags=["vectors"])

collection_service = CollectionService()

@vector_router.post("/", response_model=Vector, status_code=201)
def add(
    collection_name: str,
    vector: VectorCreate,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer: Annotated[VamanaIndexer, Depends(get_indexer)],
) -> Vector:
    vector_db = collection_service.add_vector(collection_name, vector, uow)
    indexer.update(VectorLite.from_vector(vector_db))
    return vector_db


@vector_router.get("/", response_model=Vector)
def get_by_id(
    collection_name: str, vector_id: int, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> Vector:
    return collection_service.get_vector(collection_name, vector_id, uow)


@vector_router.get("/all", response_model=List[Vector])
def get_all(
    collection_name: str, uow: Annotated[UnitOfWork, Depends(get_uow)]
) -> List[Vector]:
    return collection_service.get_all_vectors(collection_name, uow)


@vector_router.delete("/", response_model=Vector, status_code=201)
def delete_by_id(
    collection_name: str,
    vector_id: int,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
) -> Vector:
    return collection_service.delete_vector(collection_name, vector_id, uow)