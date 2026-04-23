from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import src.engine.indexers.vamana.ops as operations
import src.engine.indexers.vamana.reranking as reranking
from src.common import MetricType, config
from src.engine.indexers.vamana.controller import AlphaController
from src.engine.indexers.vamana.graph import Graph
from src.engine.indexers.vamana.vector_store import VectorStore
from src.schemas.vector import VectorData


@dataclass(frozen=True, order=True)
class VamanaConfig:
    dims: int
    metric: MetricType
    L: int
    R: int
    alpha_first_pass: float
    alpha_second_pass: float
    target_utilization: float

    def __post_init__(self):
        if not (0 < self.target_utilization <= 1):
            raise ValueError(
                f"Target utilization must be between 0 and 1. Got {self.target_utilization}."
            )
        if not (1 <= self.alpha_first_pass < 2):
            raise ValueError(
                f"Alpha for first pass must be between 1 and 2. Got {self.alpha_first_pass}."
            )
        if not (1 <= self.alpha_second_pass < 2):
            raise ValueError(
                f"Alpha for second pass must be between 0 and 1. Got {self.alpha_second_pass}."
            )
        if not (self.L >= self.R):
            raise ValueError(
                f"Search list size (L) must be larger maximum degree (R). Got L={self.L}, R={self.R}"
            )


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
        self._L: int = config.L
        self._R: int = config.R
        self._metric: int = int(config.metric)
        self._alpha_first_pass = config.alpha_first_pass
        self._alpha_second_pass = config.alpha_second_pass
        self._target_degree = self._R * config.target_utilization

        k_plant = 0.8 * self._R
        kp = 0.15 / k_plant
        ki = kp / 10.0

        self.alpha_controller = AlphaController(
            target_degree=self._target_degree,
            kp=kp,
            ki=ki,
            alpha_init=self._alpha_second_pass,
        )

    def index(self) -> None:
        self.graph = Graph.random_regular(size=self.vector_store.size, degree=self._R)
        self.entry_point = self._get_medoid()

        self._indexing_pass(alpha=self._alpha_first_pass)

        self._indexing_pass(alpha=self._alpha_second_pass)

        self.alpha_controller.reset()

    def update(self, vectors: List[VectorData] | List[int]) -> Tuple[List[int], bool]:
        if not vectors:
            return [], False

        if isinstance(vectors[0], int):
            batch_ids = vectors
        else:
            batch_ids = self.vector_store.upsert_batch(vectors)  # type: ignore

            if len(batch_ids) == 0:
                return [], False

        should_rebuild = self.entry_point is None

        if should_rebuild:
            self.index()

            return [], True
        else:
            self.graph.resize(new_size=self.vector_store.size)

            alpha = self.alpha_controller.get_alpha()

            forward_ids, backward_ids = self._index_batch(
                batch_ids=np.array(batch_ids, dtype=np.int32),
                alpha=alpha,
                return_mod_ids=True,
            )

            modified_ids = list(set(forward_ids + backward_ids))

            if backward_ids:
                total_edges = sum(
                    np.count_nonzero(self.graph.graph[node_id] != -1)
                    for node_id in backward_ids
                )
                avg_degree = total_edges / len(backward_ids)

                self.alpha_controller.feedback(avg_degree)

            return modified_ids, False

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        L_search: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
        mmr_n: Optional[int] = None,
    ) -> List[Tuple[float, int]]:
        if self.entry_point is None:
            return []

        use_mmr = mmr_lambda is not None and mmr_n is not None and mmr_n > k
        search_k = mmr_n if use_mmr else k
        search_L = L_search or max(k * 2, config.MIN_L_SEARCH)

        # normalize if using cosine similiarity
        if self._metric == int(MetricType.COSINE):
            query_vector = query_vector / np.linalg.norm(query_vector)

        (
            results,
            dists,
            _,
        ) = self._greedy_search(
            entry_id=self.entry_point,
            query_vector=query_vector,
            k=search_k,  # type: ignore
            L=search_L,
        )

        if results.size == 0:
            return []

        if mmr_lambda is not None:
            results, scores = reranking.mmr_rerank(
                query_dists=dists,
                candidate_ids=results,
                vectors=self.vector_store.vectors,
                metric=self._metric,
                k=k,
                mmr_lambda=mmr_lambda,
            )
        else:
            scores = reranking.dists_to_sims(dists=dists, metric=self._metric)

        query_results = [
            (score, self.vector_store.get_dbid(idx))
            for score, idx in zip(scores, results)
        ]

        return query_results

    def delete(self, batch_ids: List[int]) -> Tuple[List[int], bool]:
        ids_to_delete = [
            self.vector_store.dbid_to_idx[db_id]
            for db_id in batch_ids
            if db_id in self.vector_store.dbid_to_idx
        ]

        if not ids_to_delete:
            return [], False

        self.vector_store.delete_batch(batch_ids)

        alpha = self.alpha_controller.get_alpha()

        modified_ids_np = operations.delete_pass(
            alpha=alpha,
            R=self._R,
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            active_count=self.vector_store.total_size,
            metric=self._metric,
            deleted=self.vector_store.deleted,
        )

        for idx in ids_to_delete:
            self.graph.graph[idx][:] = -1

        entry_point_modified = False
        if self.entry_point in ids_to_delete:
            entry_point_modified = True
            if self.vector_store.size > 0:
                self.entry_point = self._get_medoid()
            else:
                self.entry_point = None

        modified_ids = modified_ids_np.tolist()
        return modified_ids, entry_point_modified

    def _greedy_search(
        self, entry_id: int, query_vector: np.ndarray, k: int, L: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A helper wrapper for greedy search
        """

        bitset_size = (self.vector_store.size + 7) // 8
        seen = np.zeros(bitset_size, dtype=np.uint8)
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
            deleted=self.vector_store.deleted,
        )

    def _index_batch(
        self, batch_ids: np.ndarray, alpha: float, return_mod_ids: bool = False
    ) -> Tuple[List[int], List[int]]:
        operations.forward_indexing_pass(
            batch_ids=batch_ids,
            alpha=alpha,
            R=self._R,
            L=self._L,
            entry_point=self.entry_point,  # type: ignore
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
            deleted=self.vector_store.deleted,
        )

        backward_ids_np = operations.backward_indexing_pass(
            batch_ids=batch_ids,
            alpha=alpha,
            R=self._R,
            graph=self.graph.graph,
            vectors=self.vector_store.vectors,
            metric=self._metric,
            deleted=self.vector_store.deleted,
        )

        if return_mod_ids:
            return batch_ids.tolist(), backward_ids_np.tolist()
        else:
            return [], []

    def _indexing_pass(self, alpha: float) -> None:
        sigma = self.vector_store.get_idxs()
        np.random.shuffle(sigma)

        CHUNK_SIZE = 1024
        for i in range(0, len(sigma), CHUNK_SIZE):
            chunk = sigma[i : i + CHUNK_SIZE]
            self._index_batch(batch_ids=chunk, alpha=alpha)

    def _get_medoid(self) -> int:
        sample_vectors, sample_ids = self.vector_store.get_random_sample(
            config.INDEX_RND_SAMPLE_SIZE
        )

        if self._metric == int(MetricType.COSINE):
            normed = sample_vectors / np.linalg.norm(
                sample_vectors, axis=1, keepdims=True
            )
            centroid = np.mean(normed, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            dists = 1 - normed @ centroid
        else:
            centroid = np.mean(sample_vectors, axis=0)
            dists = np.linalg.norm(sample_vectors - centroid, axis=1)

        medoid = int(np.argmin(dists))
        medoid_idx = sample_ids[medoid]

        return medoid_idx
