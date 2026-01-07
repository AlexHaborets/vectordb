from typing import List

import numpy as np

from src.common import MetricType
from src.common.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError, VectorNotFoundError, WrongVectorDimensionsError
from src.db import UnitOfWork
from src.engine import IndexerManager
from src.schemas import Collection, CollectionCreate, Vector, VectorCreate
from sklearn.preprocessing import normalize

class CollectionService():
    def __init__(self) -> None:
        pass    

    def create_collection(self, collection_data: CollectionCreate, uow: UnitOfWork) -> Collection:
        collection = uow.collections.create_collection(collection_data)
        if not collection:
            raise CollectionAlreadyExistsError(collection_data.name)
        return Collection.model_validate(collection)
    
    def delete_collection(self, collection_name: str, indexer_manager: IndexerManager, uow: UnitOfWork) -> None:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)
        uow.collections.delete_collection(collection_id=collection.id)

        indexer_manager.remove_indexer(collection_name)
    
    def get_all_collections(self, uow: UnitOfWork) -> List[Collection]:
        collections = uow.collections.get_all_collections()
        return [Collection.model_validate(c) for c in collections]
    
    def get_collection_by_name(self, collection_name: str, uow: UnitOfWork) -> Collection:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)
        return Collection.model_validate(collection)
    
    def upsert_vectors(
        self, 
        collection_name: str, 
        vectors: List[VectorCreate], 
        uow: UnitOfWork
    ) -> List[Vector]:
        collection = self.get_collection_by_name(collection_name, uow)
        
        for vec in vectors:
            dim = len(vec.vector)
            if dim != collection.dimension:
                raise WrongVectorDimensionsError(dim, collection.dimension)
        
        # Normalize vectors is the collection metric is cosine
        if collection.metric == MetricType.COSINE:
            matrix = np.array([v.vector for v in vectors])
            normalized_vectors = normalize(matrix, norm='l2')
            for i in range(len(vectors)):
                vectors[i].vector = normalized_vectors[i].tolist()
        
        new_vectors = uow.vectors.upsert_vectors(collection.id, vectors)

        result = [Vector.model_validate(v) for v in new_vectors]
        return result
    
    def get_vector(self, collection_name: str, vector_id: str, uow: UnitOfWork) -> Vector:
        collection = self.get_collection_by_name(collection_name, uow)

        vector = uow.vectors.get_vector_by_id(collection.id, vector_id)
        if not vector:
            raise VectorNotFoundError(vector_id)
        return Vector.model_validate(vector)
    
    def get_all_vectors(self, collection_name: str, uow: UnitOfWork) -> List[Vector]:
        collection = self.get_collection_by_name(collection_name, uow)

        vectors = uow.vectors.get_all_vectors(collection.id)
        
        return [Vector.model_validate(v) for v in vectors]

    def delete_vector(self, collection_name: str, vector_id: str, uow: UnitOfWork) -> None:
        collection = self.get_collection_by_name(collection_name, uow)

        success = uow.vectors.mark_vector_deleted(collection.id, vector_id)
        if not success:
            raise VectorNotFoundError(vector_id)