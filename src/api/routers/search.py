from typing import Annotated, Dict, List

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src.common import config
from src.api.dependencies import get_indexer
from src.engine import VamanaIndexer
from src.schemas import Query, SearchResult, Vector

search_router = APIRouter(prefix="/{collection_name}/search", tags=["search"])


@search_router.post("/", response_model=List[SearchResult])
def search(
    collection_name: str,
    q: Query,
    k: int,
    db: Annotated[Session, Depends(get_db_session)],
    indexer: Annotated[VamanaIndexer, Depends(get_indexer)],
) -> List[SearchResult]:
    query_results = indexer.search(q.numpy_vector, k)
    vectors = VectorDBRepository(db).get_vectors_by_ids(
        [result[1] for result in query_results]
    )
    id_to_vector: Dict[int, Vector] = {v.id: v for v in vectors}

    results: List[SearchResult] = []
    for score, vector_id in query_results:
        results.append(
            SearchResult(
                score=round(score, config.SIMILARITY_SCORE_PRECISION),
                vector=id_to_vector[vector_id],
            )
        )
    return results
