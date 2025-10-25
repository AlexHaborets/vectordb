from typing import Annotated, List

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src import config
from src.api.dependencies import get_db_session, get_indexer
from src.api.exceptions import VectorNotFoundError, WrongVectorDimensionsError
from src.core import VamanaIndexer
from src.db.crud import VectorDBRepository
from src.schemas import SearchResult, Vector, VectorCreate, Query, VectorLite

vector_router = APIRouter(prefix="/vectors", tags=["vectors"])

search_router = APIRouter(prefix="/search", tags=["vectors"])


# CREATE
@vector_router.post("/", response_model=Vector, status_code=201)
def add(
    request: Request,
    vector: VectorCreate,
    db: Annotated[Session, Depends(get_db_session)],
    indexer: Annotated[VamanaIndexer, Depends(get_indexer)],
) -> Vector:
    if dims := len(vector.vector) != config.VECTOR_DIMENSIONS:
        raise WrongVectorDimensionsError(dims)
    repo = VectorDBRepository(db)
    result = repo.add_vector(vector)
    request.app.state.logger.info("starting indexing")
    indexer.index(
        alpha=config.VAMANA_ALPHA, L=config.VAMANA_L, R=config.VAMANA_R, repo=repo
    )
    request.app.state.logger.info("finished indexing")
    return result


# READ
@vector_router.get("/", response_model=Vector)
def get_by_id(
    vector_id: int, db: Annotated[Session, Depends(get_db_session)]
) -> Vector:
    vector = VectorDBRepository(db).get_vector_by_id(vector_id)
    if not vector:
        raise VectorNotFoundError(vector_id)
    return vector


@vector_router.get("/all", response_model=List[Vector])
def get_all(db: Annotated[Session, Depends(get_db_session)]) -> List[Vector]:
    return VectorDBRepository(db).get_all_vectors()


# DELETE
@vector_router.delete("/", response_model=Vector, status_code=201)
def delete_by_id(
    vector_id: int, db: Annotated[Session, Depends(get_db_session)]
) -> Vector:
    vector = VectorDBRepository(db).mark_vector_deleted(vector_id)
    if not vector:
        raise VectorNotFoundError(vector_id)
    return vector


@search_router.post("/", response_model=List[SearchResult])
def search(
    q: Query,
    k : int,
    db: Annotated[Session, Depends(get_db_session)],
    indexer: Annotated[VamanaIndexer, Depends(get_indexer)],
) -> List[SearchResult]:
    results =  indexer.search(q, k, config.VAMANA_L, VectorDBRepository(db))
    return [SearchResult.from_vector_lite(v) for v in results]
