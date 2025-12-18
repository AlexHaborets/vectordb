import bisect
import heapq
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

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
        self._metric: MetricType = config.metric

    def _compute_dists_batch(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        if self._metric == MetricType.L2:
            return np.linalg.norm(vectors - query, axis=1)
        else:
            return 1 - np.dot(vectors, query)

    def greedy_search(
        self, entry_id: int, query_vector: np.ndarray, k: int, L: int
    ) -> tuple[list[int], set[Any]]:
        """
        Data: Graph G with start node s, query xq, result
            size k, search list size L ≥ k
        Result: Result set L containing k-approx NNs, and
            a set V containing all the visited nodes
        """
        entry_vector = self.vector_store.get(entry_id)
        query_entry_dist = self._compute_dists_batch(
            query=query_vector, vectors=entry_vector
        )

        candidates = [(query_entry_dist, entry_id)]

        visited = set()

        seen = {entry_id}

        best = [(query_entry_dist, entry_id)]

        while candidates:
            p_star_dist, p_star = heapq.heappop(candidates)  # type: ignore

            if len(best) >= L and p_star_dist > best[-1][0]:
                break

            if p_star in visited:
                continue

            visited.add(p_star)

            neighbors = self.graph.get_neighbors(p_star)

            unseen_neighbors = [n for n in neighbors if n not in seen]

            if not unseen_neighbors:
                continue
            
            unseen_neighbors_vectors = self.vector_store.get_batch(unseen_neighbors)
            dists = self._compute_dists_batch(query_vector, unseen_neighbors_vectors)

            for id, dist in zip(unseen_neighbors, dists):
                seen.add(id)
                if len(best) >= L:
                    if dist >= best[-1][0]:
                        continue
                    bisect.insort(best, (dist, id))
                    best.pop()
                else:
                    bisect.insort(best, (dist, id))

                heapq.heappush(candidates, (dist, id))  # type: ignore

        return ([x[1] for x in best[:k]], visited)

    def robust_prune(self, source: int, candidates: set[int], alpha: float) -> None:
        """
        Data: Graph G, point p ∈ P , candidate set V,
            distance threshold α ≥ 1, degree bound R
        Result: G is modified by setting at most R new
            out-neighbors for p
        """
        candidates.update(self.graph.get_neighbors(source))
        candidates.discard(source)

        if not candidates:
            self.graph.set_neighbors(source, set())
            return

        candidate_ids = list(candidates)
        candidate_vectors = self.vector_store.get_batch(candidate_ids)
        source_vector = self.vector_store.get(source)
        candidate_source_dists = self._compute_dists_batch(
            source_vector, candidate_vectors
        )

        candidates_sorted = sorted(
            zip(candidate_source_dists, candidate_ids), key=lambda x: x[0]
        )

        neighbors: List[int] = []
        neighbors_vectors: List[np.ndarray] = []

        for p_star_dist, p_star in candidates_sorted:
            if len(neighbors) >= self._R:
                break

            keep = True

            if neighbors:
                p_star_vec = self.vector_store.get(p_star)
                p_star_neighbors = np.array(neighbors_vectors)

                neighbors_dists = self._compute_dists_batch(
                    p_star_vec, p_star_neighbors
                )

                values = alpha * neighbors_dists

                if np.any(values <= p_star_dist):
                    keep = False

            if keep:
                neighbors.append(p_star)
                neighbors_vectors.append(self.vector_store.get(p_star))

        self.graph.set_neighbors(source, set(neighbors))

    def _indexing_pass(self, alpha: float) -> None:
        sigma = self.vector_store.get_idxs()
        random.shuffle(sigma)

        for i in sigma:
            query_vector = self.vector_store.get(i)
            (_, V) = self.greedy_search(
                entry_id=self.entry_point,  # type: ignore
                query_vector=query_vector,
                k=1,
                L=self._L_build,
            )

            self.robust_prune(source=i, candidates=V, alpha=alpha)

            query_neighbors = self.graph.get_neighbors(i)

            for other in query_neighbors:
                other_neighbors = set(self.graph.get_neighbors(other))
                other_neighbors.add(i)

                if len(other_neighbors) > self._R:
                    self.robust_prune(
                        source=other, candidates=other_neighbors, alpha=alpha
                    )
                else:
                    self.graph.set_neighbors(other, other_neighbors)

    def index(self) -> None:
        """
        Data: Database P with n points where i-th point has coords xi, parameters α, L, R
        Result: Directed graph G over P with out-degree <=R
        """
        self.graph = Graph.random_regular(size=self.vector_store.size(), degree=self._R)
        self.entry_point = self.get_mediod()

        self._indexing_pass(alpha=1.0)

        self._indexing_pass(alpha=1.2)

    def update(self, vectors: List[VectorData]) -> None:
        self.vector_store.add_batch(vectors)
        # TODO: Use incremental updates instead of a full reindexing
        self.index()

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, int]]:
        if not self.entry_point:
            return []

        # TODO: In the future use Beam search instead
        results, _ = self.greedy_search(
            entry_id=self.entry_point, query_vector=query_vector, k=k, L=self._L_search
        )

        if not results:
            return []

        vectors = self.vector_store.get_batch(results)
        dists = self._compute_dists_batch(query_vector, vectors)

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
        sample = self.vector_store.get_random_sample(config.INDEX_RND_SAMPLE_SIZE)

        centroid = np.mean(sample, axis=0)

        dists = np.linalg.norm(sample - centroid, axis=1)

        mediod = int(np.argmin(dists))

        return mediod
