from typing import Dict, List
from src import schemas
from src.common import config
from src.common.exceptions import CollectionNotFoundError
from src.db import UnitOfWork
from src.engine.indexer_manager import IndexerManager
from src.schemas import Query, SearchResult, Vector


class IndexService:
    def __init__(self) -> None:
        pass

    def search(
        self,
        collection_name: str,
        query: Query,
        indexer_manager: IndexerManager,
        uow: UnitOfWork,
    ) -> List[SearchResult]:
        indexer = indexer_manager.get_indexer(collection_name, uow)
        query_results = indexer.search(query.numpy_vector, query.k)
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)
        vectors = uow.vectors.get_vectors_by_ids(
            collection_id=collection.id, ids=[result[1] for result in query_results]
        )
        id_to_vector: Dict[int, schemas.Vector] = {
            v.id: schemas.Vector.model_validate(v) for v in vectors
        }

        results: List[SearchResult] = []
        for score, vector_id in query_results:
            results.append(
                SearchResult(
                    score=round(score, config.SIMILARITY_SCORE_PRECISION),
                    vector=id_to_vector[vector_id],
                )
            )
        return results

    def update(
        self,
        collection_name: str,
        vectors: List[Vector],
        indexer_manager: IndexerManager,
        uow: UnitOfWork,
    ) -> None:
        indexer_manager.update(collection_name, vectors, uow)
        
