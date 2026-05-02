from typing import Dict, List

import numpy as np

from trovadb.server.common import MetricType, VectorNotFoundError, config
from trovadb.server.common.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    WrongVectorDimensionsError,
)
from trovadb.server.db import UnitOfWork
from trovadb.server.engine import IndexerManager
from trovadb.server.schemas import (
    Collection,
    CollectionCreate,
    Query,
    SearchResult,
    Vector,
    VectorCreate,
)


class CollectionService:
    def __init__(self) -> None:
        pass

    def create_collection(
        self, collection_data: CollectionCreate, uow: UnitOfWork
    ) -> Collection:
        collection = uow.collections.create_collection(collection_data)
        if not collection:
            raise CollectionAlreadyExistsError(collection_data.name)
        return Collection.model_validate(collection)

    def delete_collection(
        self, collection_name: str, indexer_manager: IndexerManager, uow: UnitOfWork
    ) -> None:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)
        uow.collections.delete_collection(collection_id=collection.id)

        indexer_manager.remove_indexer(collection_name)

    def get_all_collections(self, uow: UnitOfWork) -> List[Collection]:
        collections = uow.collections.get_all_collections()
        return [Collection.model_validate(c) for c in collections]

    def get_collection_by_name(
        self, collection_name: str, uow: UnitOfWork
    ) -> Collection:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)
        return Collection.model_validate(collection)

    def get_vectors_by_external_id(
        self, collection_name: str, vector_ids: List[str], uow: UnitOfWork
    ) -> List[Vector]:
        collection = self.get_collection_by_name(collection_name, uow)

        vectors = uow.vectors.get_vectors_by_external_id(collection.id, vector_ids)
        return [Vector.model_validate(v) for v in vectors]

    def get_vector_by_external_id(
        self, collection_name: str, vector_id: str, uow: UnitOfWork
    ) -> Vector:
        collection = self.get_collection_by_name(collection_name, uow)

        vector = uow.vectors.get_vector_by_external_id(collection.id, vector_id)
        if not vector:
            raise VectorNotFoundError(vector_id)
        return Vector.model_validate(vector)

    def get_all_vectors(self, collection_name: str, uow: UnitOfWork) -> List[Vector]:
        collection = self.get_collection_by_name(collection_name, uow)

        vectors = uow.vectors.get_all_vectors(collection.id)

        return [Vector.model_validate(v) for v in vectors]

    def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[VectorCreate],
        indexer_manager: IndexerManager,
        uow: UnitOfWork,
    ) -> List[Vector]:
        collection = self.get_collection_by_name(collection_name, uow)

        for vec in vectors:
            dim = len(vec.vector)
            if dim != collection.dimension:
                raise WrongVectorDimensionsError(dim, collection.dimension)

        # Normalize vectors is the collection metric is cosine
        if collection.metric == MetricType.COSINE:
            matrix = np.array([v.vector for v in vectors])
            normalized_vectors = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
            for i in range(len(vectors)):
                vectors[i].vector = normalized_vectors[i].tolist()  # type: ignore

        new_vectors = uow.vectors.upsert_vectors(collection.id, vectors)
        uow.indexes.bump_version(collection.id)

        result = [Vector.model_validate(v) for v in new_vectors]

        indexer_manager.update(collection_name, result, uow)

        return result

    def delete_vectors(
        self,
        collection_name: str,
        vector_ids: List[str],
        indexer_manager: IndexerManager,
        uow: UnitOfWork,
    ) -> None:
        if not vector_ids:
            return

        collection = self.get_collection_by_name(collection_name, uow)

        deleted_vectors = uow.vectors.delete_vectors(collection.id, vector_ids)
        uow.indexes.bump_version(collection.id)

        internal_ids = [v.id for v in deleted_vectors]

        indexer_manager.delete(collection_name, internal_ids, uow)

    def search(
        self,
        collection_name: str,
        query: Query,
        indexer_manager: IndexerManager,
        uow: UnitOfWork,
    ) -> List[SearchResult]:
        with uow:
            collection = uow.collections.get_collection_by_name(collection_name)
            if not collection:
                raise CollectionNotFoundError(collection_name)

            query_results = indexer_manager.search(
                collection_name=collection_name, query=query, uow=uow
            )

            vectors = uow.vectors.get_vectors_by_id(
                collection_id=collection.id,
                vector_ids=[result[1] for result in query_results],
            )

        id_to_vector: Dict[int, Vector] = {
            v.id: Vector.model_validate(v) for v in vectors
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
