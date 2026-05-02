import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

import src.common.config as config
from src.common.exceptions import CollectionNotFoundError
from src.db import UnitOfWork, session_manager
from src.db.uow import DBUnitOfWork
from src.engine.indexers.vamana.graph import Graph
from src.engine.indexers.vamana.indexer import VamanaConfig, VamanaIndexer
from src.engine.indexers.vamana.vector_store import VectorStore
from src.engine.snapshot import IndexSnapshot, pack_snapshot, unpack_snapshot
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
            self._persist_event.wait(timeout=config.PERSIST_PERIOD)
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
        logger.info(f"Saving {collection_name} index snapshot...")

        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            return

        snapshot = self._capture_snapshot(collection_name)
        if not snapshot:
            return

        payload = pack_snapshot(snapshot)

        version = uow.indexes.get_version(collection.id)

        uow.indexes.save_snapshot(
            collection_id=collection.id,
            version=version,
            entry_point_id=snapshot.entry_point,
            payload=payload,
        )

    def _capture_snapshot(self, collection_name: str) -> Optional[IndexSnapshot]:
        with self._lock:
            if collection_name not in self._indexers_locks:
                return None
            indexer_lock = self._indexers_locks[collection_name]

        with indexer_lock:
            if collection_name not in self._indexers:
                return None

            indexer = self._indexers[collection_name]
            return indexer.get_snapshot()

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

    def _restore_indexer(
        self,
        snapshot: IndexSnapshot,
        vectors_in_db,
        vamana_config: VamanaConfig,
    ):
        dbid_to_vec = {v.id: VectorData.model_validate(v) for v in vectors_in_db}
        vectors = [dbid_to_vec[int(dbid)] for dbid in snapshot.id_map]
        vector_store = VectorStore(dims=vamana_config.dims, vectors=vectors)
        graph = Graph.from_snapshot(snapshot.graph, degree=vamana_config.R)

        return VamanaIndexer(
            config=vamana_config,
            vector_store=vector_store,
            graph=graph,
            entry_point=snapshot.entry_point,
        )

    def _repair_indexer(
        self,
        snapshot: IndexSnapshot,
        vectors_in_db,
        vamana_config: VamanaConfig,
    ) -> VamanaIndexer:
        graph = snapshot.graph
        id_map = snapshot.id_map
        entry_point = snapshot.entry_point

        dbid_to_vec = {v.id: VectorData.model_validate(v) for v in vectors_in_db}
        db_ids = set(dbid_to_vec.keys())
        snapshot_ids = set(snapshot.id_map.tolist())

        orphaned = snapshot_ids - db_ids
        unindexed_ids = db_ids - snapshot_ids

        # First pass: remove orphaned vectors
        if orphaned:
            keep = np.array([int(dbid) in db_ids for dbid in id_map])
            valid_indices = np.where(keep)[0].astype(np.int32)

            remap = np.full(len(id_map), -1, dtype=np.int32)
            remap[valid_indices] = np.arange(len(valid_indices), dtype=np.int32)

            graph = graph[valid_indices].copy()
            valid_edges = graph != -1
            graph[valid_edges] = remap[graph[valid_edges]]

            id_map = id_map[valid_indices]
            entry_point = int(remap[entry_point]) if remap[entry_point] != -1 else None

        vectors = [dbid_to_vec[int(dbid)] for dbid in id_map]
        vector_store = VectorStore(dims=vamana_config.dims, vectors=vectors)

        graph = Graph.from_snapshot(graph, degree=vamana_config.R)

        indexer = VamanaIndexer(
            config=vamana_config,
            vector_store=vector_store,
            graph=graph,
            entry_point=entry_point,
        )

        # Second pass
        if unindexed_ids:
            indexer.update([dbid_to_vec[dbid] for dbid in unindexed_ids])

        return indexer

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

        db_snapshot = uow.indexes.get_snapshot(collection_id=collection.id)
        if not db_snapshot:
            vectors = [VectorData.model_validate(v) for v in vectors_in_db]
            vector_store = VectorStore(dims=vamana_config.dims, vectors=vectors)
            return VamanaIndexer(config=vamana_config, vector_store=vector_store)

        snapshot = unpack_snapshot(db_snapshot.payload, db_snapshot.entry_point_id)
        current_version = uow.indexes.get_version(collection.id)
        if db_snapshot.version == current_version:
            return self._restore_indexer(
                snapshot=snapshot,
                vectors_in_db=vectors_in_db,
                vamana_config=vamana_config,
            )

        logger.warning(f"repairing index for {collection_name}")
        return self._repair_indexer(
            snapshot=snapshot, vectors_in_db=vectors_in_db, vamana_config=vamana_config
        )

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
