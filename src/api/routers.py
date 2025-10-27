from typing import Annotated, Dict, List

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src import config
from src.api.dependencies import get_db_session, get_indexer
from src.api.exceptions import VectorNotFoundError, WrongVectorDimensionsError
from src.core import VamanaIndexer
from src.db.crud import VectorDBRepository
from src.schemas import Query, SearchResult, Vector, VectorCreate, VectorLite

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
    
    new_vector = VectorDBRepository(db).add_vector(vector)
    request.app.state.logger.info("starting indexing")
    indexer.update(vector=VectorLite.from_vector(new_vector))
    request.app.state.logger.info("finished indexing")
    return new_vector


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
    k: int,
    db: Annotated[Session, Depends(get_db_session)],
    indexer: Annotated[VamanaIndexer, Depends(get_indexer)],
) -> List[SearchResult]:
    query_results = indexer.search(q.numpy_vector, k)
    vectors = VectorDBRepository(db).get_vectors_by_ids([result[1] for result in query_results])
    id_to_vector: Dict[int, Vector] = {v.id: v for v in vectors}

    results: List[SearchResult] = []
    for score, vector_id in query_results:
        results.append(
            SearchResult(
                score=round(score, config.SIMILARITY_SCORE_PRECISION), 
                vector=id_to_vector[vector_id]
            ) 
        )
    return results

@search_router.post("/reindex")
def reindex(request: Request, indexer: Annotated[VamanaIndexer, Depends(get_indexer)]) -> Dict[str, str]:    
    request.app.state.logger.info("starting indexing")
    indexer.index()
    request.app.state.logger.info("finished indexing")
    return {"message": "Reindexed successfully"}