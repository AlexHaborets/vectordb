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
    L_search: int = 64
    R: int = 32


class VamanaIndexer:
    def __init__(
        self,
        config: VamanaConfig,
        vector_store: VectorStore,
        graph: Optional[Graph] = None,
        entry_point: Optional[int] = None,
    ) -> None:
        self.graph = graph if graph else Graph(size=vector_store.size, degree=config.R)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A helper wrapper for greedy search
        """
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
        self, source_id: int, candidates_ids: np.ndarray, alpha: float
    ) -> None:
        """
        A helper wrapper for robust prune
        """
        operations.robust_prune(
            source_id=source_id,
            candidates_ids=candidates_ids,
            alpha=alpha,
            R=self._R,
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
        )

    def _index_batch(self, batch_ids: np.ndarray, alpha: float) -> None:
        operations.forward_indexing_pass(
            batch_ids=batch_ids,
            alpha=alpha,
            R=self._R,
            L=self._L_build,
            entry_point=self.entry_point,  # type: ignore
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
        )

        operations.backward_indexing_pass(
            batch_ids=batch_ids,
            alpha=alpha,
            R=self._R,
            L=self._L_build,
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
        )

    def _indexing_pass(self, alpha: float) -> None:
        sigma = self.vector_store.get_idxs()
        np.random.shuffle(sigma)

        self._index_batch(batch_ids=sigma, alpha=alpha)

    def index(self) -> None:
        """
        Data: Database P with n points where i-th point has coords xi, parameters α, L, R
        Result: Directed graph G over P with out-degree <=R
        """
        self.graph = Graph.random_regular(size=self.vector_store.size, degree=self._R)
        self.entry_point = self.get_medoid()

        self._indexing_pass(alpha=1.0)

        self._indexing_pass(alpha=1.2)

    def update(self, vectors: List[VectorData] | List[int]) -> None:
        if not vectors:
            return
        
        if isinstance(vectors[0], int):
            batch_ids = vectors
        else:
            batch_ids = self.vector_store.upsert_batch(vectors) # type: ignore

            if batch_ids.size == 0:
                return

        curr_size = self.vector_store.size

        should_rebuild = (
            self.entry_point is None or curr_size < 10000 and len(batch_ids) > 1000
        )

        if should_rebuild:
            self.index()
        else:
            self.graph.resize(new_size=self.vector_store.size)

            self._index_batch(batch_ids=np.array(batch_ids), alpha=1.2)

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

    def get_medoid(self) -> int:
        sample_vectors, ids = self.vector_store.get_random_sample(
            config.INDEX_RND_SAMPLE_SIZE
        )

        centroid = np.mean(sample_vectors, axis=0)

        dists = np.linalg.norm(sample_vectors - centroid, axis=1)

        medoid = int(np.argmin(dists))

        medoid_idx = ids[medoid]

        return medoid_idx
