from typing import Dict
from src.common.exceptions import CollectionNotFoundError
from src.db import UnitOfWork
from src.engine import VamanaConfig, VamanaIndexer
import src.common.config as config
from src.engine.structures.graph import Graph
from src.engine.structures.vector_store import VectorStore
from src.schemas import VectorLite

class IndexerManager:
    def __init__(self) -> None:
        self._indexers: Dict[str, VamanaIndexer]= {}

    def get_indexer(self, collection_name: str, uow: UnitOfWork) -> VamanaIndexer:
        if collection_name in self._indexers:
            return self._indexers[collection_name] 
        
        indexer = self._load_from_db(collection_name, uow)
        self._indexers[collection_name] = indexer
        return indexer

    def remove_indexer(self, collection_name: str) -> None:
        if collection_name in self._indexers:
            del self._indexers[collection_name]
        
    def _load_from_db(self,collection_name: str, uow: UnitOfWork) -> VamanaIndexer:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)
        
        graph_in_db = uow.vectors.get_graph(collection_id=collection.id)
        graph = Graph(graph=graph_in_db) if graph_in_db else None 
        
        vamana_config = VamanaConfig(
            metric=collection.metric,
            dims=collection.dimension,
            alpha=config.VAMANA_ALPHA,
            L_build=config.VAMANA_L_BUILD,
            L_search=config.VAMANA_L_SEARCH,
            R=config.VAMANA_R
        ) 
        
        vectors_in_db = uow.vectors.get_all_vectors(collection.id)
        vectors = [VectorLite.model_validate(v) for v in vectors_in_db]
        vector_store = VectorStore.build_from_vectors(vectors=vectors, dims=collection.dimension) 

        entry_point_in_db = uow.collections.get_index_metadata(
            collection_id=collection.id, 
            key="entry_point"
        ) 
        entry_point = int(entry_point_in_db) if entry_point_in_db else None

        indexer = VamanaIndexer(
            config=vamana_config,
            vector_store=vector_store,
            graph=graph,
            entry_point=entry_point
        )
        return indexer
    
    def _save_to_db(self, collection_name: str, uow: UnitOfWork) -> None:
        if collection_name not in self._indexers:
            return
        
        indexer = self._indexers[collection_name]
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)

        uow.vectors.save_graph(
            collection_id=collection.id,
            graph=indexer.graph.graph
        )

        if indexer.entry_point:
            uow.collections.set_index_metadata(
                collection_id=collection.id,
                key="entry_point",
                value=str(indexer.entry_point)
            )

    def save_all(self, uow: UnitOfWork):
        for collection in self._indexers.keys():
            self._save_to_db(collection, uow)
