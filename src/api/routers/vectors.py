from typing import Annotated, List

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src import config
from src.api.dependencies import get_db_session, get_indexer
from src.api.exceptions import VectorNotFoundError, WrongVectorDimensionsError
from src.core import VamanaIndexer
from src.db.crud import VectorDBRepository
from src.schemas import Vector, VectorCreate, VectorLite

vector_router = APIRouter(prefix="/{collection_name}/vectors", tags=["vectors"])


@vector_router.post("/", response_model=Vector, status_code=201)
def add(
    request: Request,
    collection_name: str,
    vector: VectorCreate,
    db: Annotated[Session, Depends(get_db_session)],
    indexer: Annotated[VamanaIndexer, Depends(get_indexer)],
) -> Vector:
    if dims := len(vector.vector) != config.VECTOR_DIMENSIONS:
        raise WrongVectorDimensionsError(dims)

    new_vector = VectorDBRepository(db).add_vector(vector)
    indexer.update(vector=VectorLite.from_vector(new_vector))
    return new_vector


@vector_router.get("/", response_model=Vector)
def get_by_id(
    collection_name: str,
    vector_id: int, 
    db: Annotated[Session, Depends(get_db_session)]
) -> Vector:
    vector = VectorDBRepository(db).get_vector_by_id(vector_id)
    if not vector:
        raise VectorNotFoundError(vector_id)
    return vector


@vector_router.get("/all", response_model=List[Vector])
def get_all(
    collection_name: str,
    db: Annotated[Session, Depends(get_db_session)]
) -> List[Vector]:
    return VectorDBRepository(db).get_all_vectors()


@vector_router.delete("/", response_model=Vector, status_code=201)
def delete_by_id(
    collection_name: str,
    vector_id: int, 
    db: Annotated[Session, Depends(get_db_session)]
) -> Vector:
    vector = VectorDBRepository(db).mark_vector_deleted(vector_id)
    if not vector:
        raise VectorNotFoundError(vector_id)
    return vector
