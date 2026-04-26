import copy
import pickle
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

import src.common.config as config
from src.common.exceptions import CollectionNotFoundError
from src.db import UnitOfWork, session_manager
from src.db.uow import DBUnitOfWork
from src.engine import VamanaConfig, VamanaIndexer
from src.engine.indexers.vamana.graph import Graph
from src.engine.indexers.vamana.vector_store import VectorStore
from src.schemas import Query, Vector, VectorData


class IndexerManager:
    def __init__(self) -> None:
        self._indexers: Dict[str, VamanaIndexer] = {}
        self._indexers_locks: Dict[str, threading.RLock] = {}
        self._lock = threading.RLock()

        self._save_queue: Set[str] = set()
        self._save_queue_lock = threading.RLock()

        self._persist_event = threading.Event()
        self._running = True

        self._worker = threading.Thread(target=self._persist_loop)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._persist_event.set()
        self._worker.join()

    def _persist_loop(self) -> None:
        while self._running:
            self._persist_event.wait(timeout=5.0)
            self._persist_event.clear()
            self._persist_job()

        self._persist_job()

    def _persist_job(self) -> None:
        with self._save_queue_lock:
            if not self._save_queue:
                return
            save_queue_snapshot = list(self._save_queue)
            self._save_queue.clear()

        for collection_name in save_queue_snapshot:
            try:
                uow = DBUnitOfWork(session_manager.get_session_factory())
                with uow:
                    self._persist_index(collection_name, uow)
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Failed to persist index for {collection_name}: {e}")
                self._enqueue_collection(collection_name)

    def _enqueue_collection(self, collection_name: str) -> None:
        with self._save_queue_lock:
            self._save_queue.add(collection_name)

    def _persist_index(self, collection_name: str, uow: UnitOfWork) -> None:
        snapshot = self._capture_snapshot(collection_name)
        if not snapshot:
            return

        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            return

        logger.info(f"Saving {collection_name} index snapshot...")
        version = uow.indexes.get_version(collection.id)
        
        uow.indexes.save_snapshot(
            collection_id=collection.id,
            version=version,
            entry_point_id=inde,
            graph=snapshot,
        )

    def _capture_snapshot(self, collection_name: str) -> Optional[Dict[int, List[int]]]:
        with self._lock:
            if collection_name not in self._indexers_locks:
                return None
            indexer_lock = self._indexers_locks[collection_name]

        with indexer_lock:
            if collection_name not in self._indexers:
                return None

            indexer = self._indexers[collection_name]
            return indexer.graph.to_db_graph(indexer.vector_store.idx_to_dbid)

    def get_indexer(self, collection_name: str, uow: UnitOfWork) -> VamanaIndexer:
        with self._lock:
            if collection_name in self._indexers:
                return self._indexers[collection_name]

            if collection_name not in self._indexers_locks:
                self._indexers_locks[collection_name] = threading.RLock()

            indexer = self._load_from_db(collection_name, uow)
            self._indexers[collection_name] = indexer

            return indexer

    def remove_indexer(self, collection_name: str) -> None:
        with self._lock:
            if collection_name in self._indexers_locks:
                indexer_lock = self._indexers_locks[collection_name]

                # Prevent using the indexer while its being deleted
                with indexer_lock:
                    if collection_name in self._indexers:
                        del self._indexers[collection_name]

                # Delete the indexer lock
                del self._indexers_locks[collection_name]

        # Clean the queue
        with self._save_queue_lock:
            if collection_name in self._save_queue:
                self._save_queue.discard(collection_name)

    def _load_from_db(self, collection_name: str, uow: UnitOfWork) -> VamanaIndexer:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)

        vamana_config = VamanaConfig(
            metric=collection.metric,
            dims=collection.dimension,
            L=config.VAMANA_L,
            R=config.VAMANA_R,
            alpha_first_pass=config.VAMANA_ALPHA_FIRST_PASS,
            alpha_second_pass=config.VAMANA_ALPHA_SECOND_PASS,
            target_utilization=config.VAMANA_TARGET_UTILIZATION,
        )

        vectors_in_db = uow.vectors.get_all_vectors(collection.id)
        vectors = [VectorData.model_validate(v) for v in vectors_in_db]
        vector_store = VectorStore(dims=collection.dimension, vectors=vectors)

        snapshot = uow.indexes.get_snapshot(collection_id=collection.id)
        if not snapshot:
            raise ValueError("oh no")

        try:
            db_graph = pickle.loads(snapshot.payload)
        except (OSError, pickle.PickleError):
            raise ValueError("oh no")

        graph = Graph.from_db(
            db_graph=db_graph,
            degree=vamana_config.R,
            dbid_to_idx=vector_store.dbid_to_idx,
        )

        indexer = VamanaIndexer(
            config=vamana_config,
            vector_store=vector_store,
            graph=graph,
            entry_point=snapshot.entry_point_id or None,
        )

        unindexed_db_ids = uow.indexes.get_unindexed_vector_ids(
            collection_id=collection.id
        )

        # If there are vectors in db that are not in graph
        # and indexer is not new
        if unindexed_db_ids and entry_point is not None:
            logger.info(f"Repairing index for {collection_name}")

            unindexed_ids = [
                vector_store.get_idx(db_id)
                for db_id in unindexed_db_ids
                if db_id in vector_store.dbid_to_idx
            ]
            if unindexed_ids:
                indexer.update(unindexed_ids)

        return indexer

    def update(
        self, collection_name: str, vectors: List[Vector], uow: UnitOfWork
    ) -> None:
        indexer = self.get_indexer(collection_name, uow)
        with self._indexers_locks[collection_name]:
            indexer.update(vectors=[VectorData.from_vector(v) for v in vectors])

        self._enqueue_collection(collection_name)

    def delete(
        self, collection_name: str, vectors_ids: List[int], uow: UnitOfWork
    ) -> None:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)

        indexer = self.get_indexer(collection_name, uow)
        with self._indexers_locks[collection_name]:
            indexer.delete(vectors_ids)

            self._enqueue_collection(collection_name)

    def search(
        self, collection_name: str, query: Query, uow: UnitOfWork
    ) -> List[Tuple[float, int]]:
        indexer = self.get_indexer(collection_name, uow)
        with self._indexers_locks[collection_name]:
            return indexer.search(
                query.numpy_vector,
                query.k,
                query.L_search,
                query.mmr_lambda,
                query.mmr_n,
            )

    def save_all(self, uow: UnitOfWork) -> None:
        with self._lock:
            collection_names = list(self._indexers.keys())

        for collection_name in collection_names:
            collection = uow.collections.get_collection_by_name(collection_name)
            if not collection:
                continue

            collection_id = collection.id  # type: ignore
            if unindexed_db_ids := uow.indexes.get_unindexed_vector_ids(collection_id):
                indexer = self._indexers.get(collection_name)
                if indexer:
                    unindexed_idxs = [
                        indexer.vector_store.get_idx(dbid)
                        for dbid in unindexed_db_ids
                        if dbid in indexer.vector_store.dbid_to_idx
                    ]
                    if unindexed_idxs:
                        self._persist_index(collection_name, unindexed_idxs)
