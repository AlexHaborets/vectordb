from collections import defaultdict
import copy
import time
from typing import Dict, List, Tuple
import threading

from loguru import logger
from sqlalchemy.exc import IntegrityError

import src.common.config as config
from src.common.exceptions import CollectionNotFoundError
from src.db import UnitOfWork, session_manager
from src.db.uow import DBUnitOfWork
from src.engine import VamanaConfig, VamanaIndexer
from src.engine.structures.graph import Graph
from src.engine.structures.vector_store import VectorStore
from src.schemas import Query, Vector, VectorData


class IndexerManager:
    def __init__(self) -> None:
        self._indexers: Dict[str, VamanaIndexer] = {}
        self._indexers_locks: Dict[str, threading.RLock] = {}
        self._lock = threading.RLock()

        self._save_queue = defaultdict(set)
        self._save_queue_lock = threading.RLock()

        self.PERSIST_SIZE = 5000

        self._persist_event = threading.Event()
        self._running = True

        self._worker = threading.Thread(target=self._persist_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._persist_event.set()
        self._worker.join()

    def _persist_loop(self) -> None:
        while self._running:
            self._persist_event.wait(timeout=5.0)
            self._persist_event.clear()

            if not self._running:
                break

            self._persist_job()

    def _persist_job(self) -> None:
        save_queue_snapshot = {}
        with self._save_queue_lock:
            if not self._save_queue:
                return

            save_queue_snapshot = copy.deepcopy(self._save_queue)
            self._save_queue.clear()

        for collection_name, ids in save_queue_snapshot.items():
            ids_list = list(ids)

            BATCH_SIZE = 500
            for i in range(0, len(ids_list), BATCH_SIZE):
                chunk = ids_list[i : i + BATCH_SIZE]
                try:
                    self._persist_index(collection_name, chunk)

                    time.sleep(0.05)
                except Exception as e:
                    logger.error(f"Failed to persist index for {collection_name}: {e}")

                    with self._save_queue_lock:
                        self._save_queue[collection_name].update(chunk)

    def _enqueue_ids(self, collection_name: str, ids: List[int]) -> None:
        with self._save_queue_lock:
            self._save_queue[collection_name].update(ids)

            curr_size = len(self._save_queue[collection_name])

            if curr_size >= self.PERSIST_SIZE:
                self._persist_event.set()

    def _persist_index(self, collection_name: str, ids: List[int]) -> None:
        subgraph = None

        indexer_lock = None
        with self._lock:
            if collection_name in self._indexers_locks:
                indexer_lock = self._indexers_locks[collection_name]

        if not indexer_lock:
            return

        with indexer_lock:
            if collection_name not in self._indexers:
                return

            indexer = self._indexers[collection_name]

            subgraph = indexer.graph.get_subgraph(
                ids=ids, idx_to_dbid=indexer.vector_store.idx_to_dbid
            )

        if subgraph:
            try:
                uow = DBUnitOfWork(session_manager.get_session_factory())

                with uow:
                    collection = uow.collections.get_collection_by_name(collection_name)

                    logger.info(f"Updating {collection_name} index in db...")
                    uow.indexes.update_graph(
                        collection_id=collection.id,  # type: ignore
                        subgraph=subgraph,
                    )
            except IntegrityError:
                return

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
                del self._save_queue[collection_name]

    def _load_from_db(self, collection_name: str, uow: UnitOfWork) -> VamanaIndexer:
        collection = uow.collections.get_collection_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(collection_name)

        vamana_config = VamanaConfig(
            metric=collection.metric,
            dims=collection.dimension,
            L_build=config.VAMANA_L_BUILD,
            L_search=config.VAMANA_L_SEARCH,
            R=config.VAMANA_R,
            alpha_first_pass=config.VAMANA_ALPHA_FIRST_PASS,
            alpha_second_pass=config.VAMANA_ALPHA_SECOND_PASS,
        )

        vectors_in_db = uow.vectors.get_all_vectors(collection.id)
        vectors = [VectorData.model_validate(v) for v in vectors_in_db]
        vector_store = VectorStore(dims=collection.dimension, vectors=vectors)

        entry_point_in_db = uow.indexes.get_index_metadata(
            collection_id=collection.id, key="entry_point"
        )
        entry_point = (
            vector_store.get_idx(int(entry_point_in_db)) if entry_point_in_db else None
        )

        graph_in_db = uow.indexes.get_graph(collection_id=collection.id)
        graph = Graph.from_db(
            db_graph=graph_in_db,
            degree=vamana_config.R,
            dbid_to_idx=vector_store.dbid_to_idx,
        )

        indexer = VamanaIndexer(
            config=vamana_config,
            vector_store=vector_store,
            graph=graph,
            entry_point=entry_point,
        )

        unindexed_vectors = uow.indexes.get_unindexed_vector_ids(
            collection_id=collection.id
        )

        # If there are vectors in db that are not in graph
        # and indexer is not new
        if unindexed_vectors and entry_point is not None:
            logger.info(f"Repairing index for {collection_name}")
            indexer.update(unindexed_vectors)

        return indexer

    def _save_to_db(self, collection_name: str, uow: UnitOfWork) -> None:
        logger.info("Saving full index to db")
        if collection_name not in self._indexers:
            return

        graph = None
        entry_point = None
        with self._lock:
            indexer = self._indexers[collection_name]
            graph = indexer.graph.to_db_graph(
                idx_to_dbid=indexer.vector_store.idx_to_dbid
            )
            if indexer.entry_point is not None:
                entry_point = indexer.vector_store.get_dbid(indexer.entry_point)

        if not graph:
            return

        collection = uow.collections.get_collection_by_name(collection_name)

        if not collection:
            raise CollectionNotFoundError(collection_name)

        uow.indexes.save_graph(collection_id=collection.id, graph=graph)

        if entry_point is not None:
            uow.indexes.set_index_metadata(
                collection_id=collection.id,
                key="entry_point",
                value=str(entry_point),
            )

    def update(
        self, collection_name: str, vectors: List[Vector], uow: UnitOfWork
    ) -> None:
        indexer = self.get_indexer(collection_name, uow)
        modified_ids: List[int]
        full_rebuild: bool
        with self._indexers_locks[collection_name]:
            modified_ids, full_rebuild = indexer.update(
                vectors=[VectorData.from_vector(v) for v in vectors]
            )

        if full_rebuild:
            self._save_to_db(collection_name, uow)
        else:
            self._enqueue_ids(collection_name, modified_ids)

    def search(
        self, collection_name: str, query: Query, uow: UnitOfWork
    ) -> List[Tuple[float, int]]:
        indexer = self.get_indexer(collection_name, uow)
        with self._indexers_locks[collection_name]:
            return indexer.search(query.numpy_vector, query.k)

    def save_all(self, uow: UnitOfWork) -> None:
        with self._lock:
            collection_names = list(self._indexers.keys())

        for collection_name in collection_names:
            collection = uow.collections.get_collection_by_name(collection_name)
            if not collection:
                continue

            collection_id = collection.id  # type: ignore
            if ids := uow.indexes.get_unindexed_vector_ids(collection_id):
                # TODO: Use collection id directly to avoid unnecessary lookup
                self._persist_index(collection_name, ids)
