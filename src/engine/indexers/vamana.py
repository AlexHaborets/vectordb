import heapq
import random
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.common import config
from src.common.metrics import MetricType, get_metric
from src.engine.structures.graph import Graph
from src.engine.structures.vector_store import VectorStore
from src.schemas import Vector, VectorLite
from dataclasses import dataclass


@dataclass
class VamanaConfig:
    dims: int
    metric: MetricType
    alpha: float = 1.2
    L_build: int = 64
    L_search: int = 100
    R: int = 32


class VamanaIndexer:
    def __init__(
            self, 
            config: VamanaConfig, 
            vector_store: VectorStore, 
            graph: Optional[Graph] = None,
            entry_point: Optional[int] = None
    ) -> None:
        self.graph = graph if graph else Graph()
        self.entry_point = entry_point
        self.vector_store = vector_store

        # Indexing/search parameters
        self._dims = config.dims
        self._alpha = config.alpha
        self._L_build = config.L_build
        self._L_search = config.L_search
        self._R = config.R
        self._metric = MetricType(config.metric) 
        self._distance = get_metric(config.metric)

    def greedy_search(
        self, entry_id: int, query_vector: np.ndarray, k: int, L: int
    ) -> tuple[list[int], set[Any]]:
        """
        Data: Graph G with start node s, query xq, result
            size k, search list size L ≥ k
        Result: Result set L containing k-approx NNs, and
            a set V containing all the visited nodes
        """
        candidates = set([entry_id])
        visited = set()

        def distance_fn(id: int) -> float:
            p = self.vector_store.get(id)
            return self._distance(p, query_vector)

        while candidates - visited:
            p_star = min(candidates - visited, key=distance_fn)
            candidates = candidates.union(self.graph.get_neighbors(p_star))
            visited.add(p_star)

            if len(candidates) > L:
                candidates = set(heapq.nsmallest(L, candidates, key=distance_fn))

        closest_k = heapq.nsmallest(k, candidates, key=distance_fn)
        return (closest_k, visited)

    def robust_prune(self, source: int, candidates: set[int]) -> None:
        """
        Data: Graph G, point p ∈ P , candidate set V,
            distance threshold α ≥ 1, degree bound R
        Result: G is modified by setting at most R new
            out-neighbors for p
        """
        candidates = candidates.union(self.graph.get_neighbors(source))
        if source in candidates:
            candidates.remove(source)
        self.graph.set_neighbors(source, set())

        source_vector = self.vector_store.get(source)

        def distance_fn(id: int) -> float:
            p = self.vector_store.get(id)
            return self._distance(p, source_vector)

        while candidates:
            p_star = min(candidates, key=distance_fn)
            self.graph.add_neighbors(source, set([p_star]))

            if len(self.graph.get_neighbors(source)) == self._R:
                break

            to_prune = []
            p_star_vector = self.vector_store.get(p_star)
            for other in candidates:
                other_vector = self.vector_store.get(other)
                if self._alpha * self._distance(
                    p_star_vector, other_vector
                ) <= self._distance(source_vector, other_vector):
                    to_prune.append(other)

            for vector in to_prune:
                candidates.remove(vector)

    def index(self) -> None:
        """
        Data: Database P with n points where i-th point has coords xi, parameters α, L, R
        Result: Directed graph G over P with out-degree <=R
        """
        vector_ids = self.vector_store.get_dbids()
        self.graph = Graph.random_regular(verteces=vector_ids, r=self._R)

        sigma = vector_ids
        random.shuffle(sigma)
        mediod_id = self.get_mediod()
        if not mediod_id:
            raise ValueError("what")

        for i in sigma:
            query_vector = self.vector_store.get(i)
            (_, V) = self.greedy_search(
                entry_id=mediod_id, query_vector=query_vector, k=1, L=self._L_build
            )

            self.robust_prune(source=i, candidates=V)

            query_neighbors = self.graph.get_neighbors(i)
            for other in query_neighbors:
                other_neighbors = self.graph.get_neighbors(other)
                other_neighbors = set(other_neighbors + [i])

                if len(other_neighbors) > self._R:
                    self.robust_prune(source=other, candidates=other_neighbors)
                else:
                    self.graph.set_neighbors(other, other_neighbors)

        self.entry_point = mediod_id

    def update(self, vector: Vector) -> None:
        self.vector_store.add(VectorLite.from_vector(vector))
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

        def get_score(vec: np.ndarray) -> float:
            distance = self._distance(vec, query_vector)
            return 1 / (1 + distance)

        result_vectors = self.vector_store.get_batch(results)
        query_results = [
            (get_score(vector), index) for vector, index in zip(result_vectors, results)
        ]
        query_results.sort(key=lambda x: x[0], reverse=True)

        return query_results

    def get_mediod(self) -> Optional[int]:
        sample = self.vector_store.get_random_sample(config.INDEX_RND_SAMPLE_SIZE)
        if not sample:
            return None

        if len(sample) == 1:
            return list(sample.keys())[0]

        ids = list(sample.keys())
        vectors = np.array(list(sample.values()))

        # calculcate pair-wise distances for each vector
        distances = pdist(vectors, metric=self._metric.value) # type: ignore

        distance_matrix = squareform(distances)

        distance_sum = distance_matrix.sum(axis=1)

        mediod = np.argmin(distance_sum)

        medoid_id = ids[mediod]

        return medoid_id