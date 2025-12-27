from concurrent.futures import ThreadPoolExecutor
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import src.engine.indexers.vamana.ops as operations
from src.common import MetricType, config
from src.engine.structures.graph import Graph
from src.engine.structures.vector_store import VectorStore
from src.schemas.vector import VectorData


@dataclass(frozen=True, order=True)
class VamanaConfig:
    dims: int
    metric: MetricType
    L_build: int = 64
    L_search: int = 100
    R: int = 32


class VamanaIndexer:
    def __init__(
        self,
        config: VamanaConfig,
        vector_store: VectorStore,
        graph: Optional[Graph] = None,
        entry_point: Optional[int] = None,
    ) -> None:
        self.graph = graph if graph else Graph(size=config.dims, degree=config.R)
        self.entry_point = entry_point
        self.vector_store = vector_store

        # Indexing/search parameters
        self._dims: int = config.dims
        self._L_build: int = config.L_build
        self._L_search: int = config.L_search
        self._R: int = config.R
        self._metric: int = int(config.metric)

    def _greedy_search(
        self, entry_id: int, query_vector: np.ndarray, k: int, L: int, seen: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, List]:
        return operations.greedy_search(
            entry_id=entry_id,
            query_vector=query_vector,
            k=k,
            L=L,
            seen=seen,
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
        )

    def _robust_prune(
        self, source_id: int, candidates_ids: List[int], alpha: float
    ) -> None:
        neighbors = operations.robust_prune(
            source_id=source_id,
            candidates_ids=np.array(candidates_ids),
            alpha=alpha,
            R=self._R,
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
        )

        self.graph.graph[source_id] = neighbors

    def _indexing_batch(self, batch_ids: List[int], alpha: float) -> None:
        seen = np.zeros(self.vector_store.size, dtype=np.bool_)

        for i in batch_ids:
            query_vector = self.vector_store.get(i)
            _, _, V = self._greedy_search(
                entry_id=self.entry_point,  # type: ignore
                query_vector=query_vector,
                k=1,
                L=self._L_build,
                seen=seen,
            )

            self._robust_prune(source_id=i, candidates_ids=V, alpha=alpha)

            query_neighbors = self.graph.get_neighbors(i)

            for other in query_neighbors:
                with self.graph.get_lock(other):
                    other_neighbors = self.graph.get_neighbors(other)

                    if i in other_neighbors:
                        continue

                    other_neighbors.append(i) 
                    
                    if len(other_neighbors) > self._R:
                        self._robust_prune(
                            source_id=other, candidates_ids=other_neighbors, alpha=alpha
                        )
                    else:
                        self.graph.set_neighbors(other, set(other_neighbors))

    def _indexing_pass(self, alpha: float) -> None:
        sigma = self.vector_store.get_idxs()
        random.shuffle(sigma)

        num_threads = os.cpu_count() or 4
        chunk_size = max(100, len(sigma) // (num_threads * 4))
        chunks = [sigma[i : i + chunk_size] for i in range(0, len(sigma), chunk_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self._indexing_batch, chunk, alpha) for chunk in chunks
            ]
            for f in futures:
                f.result()

    def index(self) -> None:
        """
        Data: Database P with n points where i-th point has coords xi, parameters α, L, R
        Result: Directed graph G over P with out-degree <=R
        """
        self.graph = Graph.random_regular(size=self.vector_store.size, degree=self._R)
        self.entry_point = self.get_mediod()

        self._indexing_pass(alpha=1.0)

        self._indexing_pass(alpha=1.2)

    def update(self, vectors: List[VectorData]) -> None:
        self.vector_store.add_batch(vectors)
        # TODO: Use incremental updates instead of a full reindexing
        self.index()

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, int]]:
        if self.entry_point is None:
            return []

        # TODO: In the future use Beam search instead
        seen = np.zeros(self.vector_store.size, dtype=np.bool_)
        results, dists, _ = self._greedy_search(
            entry_id=self.entry_point,
            query_vector=query_vector,
            k=k,
            L=self._L_search,
            seen=seen,
        )

        if results.size == 0:
            return []

        if self._metric != MetricType.L2:
            scores = 1.0 - dists
        else:
            scores = 1 / (1 + dists)

        query_results = [
            (score, self.vector_store.get_dbid(idx))
            for score, idx in zip(scores, results)
        ]

        query_results.sort(key=lambda x: x[0], reverse=True)

        return query_results

    def get_mediod(self) -> int:
        sample_vectors, ids = self.vector_store.get_random_sample(config.INDEX_RND_SAMPLE_SIZE)

        centroid = np.mean(sample_vectors, axis=0)

        dists = np.linalg.norm(sample_vectors - centroid, axis=1)

        mediod = int(np.argmin(dists))

        mediod_idx = ids[mediod]

        return mediod_idx
