import heapq
import random
from copy import copy
from random import shuffle
from typing import Any, Dict, List, Optional

import numpy as np

from src import config
from src.db.crud import VectorDBRepository
from src.schemas import VectorLite


class VamanaIndexer:
    def __init__(self, dims: int):
        self.graph: Dict[int, List[int]] = {}
        self.entry_point: int

    def greedy_search(
        self,
        entry: VectorLite,
        query: VectorLite,
        k: int,
        L: int,
        repo: VectorDBRepository,
    ) -> tuple[list[VectorLite], set[Any]]:
        """
        Data: Graph G with start node s, query xq, result
            size k, search list size L ≥ k
        Result: Result set L containing k-approx NNs, and
            a set V containing all the visited nodes
        """
        candidates = set([entry])
        visited = set()

        def distance_fn(p: VectorLite):
            return self.distance(p, query)

        while candidates - visited:
            p_star = min(candidates - visited, key=distance_fn)
            candidates = candidates.union(self.get_neighbors(p_star, repo))
            visited.add(p_star)

            if len(candidates) > L:
                candidates = set(heapq.nsmallest(L, candidates, key=distance_fn))

        closest_k = heapq.nsmallest(k, candidates, key=distance_fn)
        return (closest_k, visited)

    def robust_prune(self, entry: VectorLite, candidates: set, alpha: float, r: int):
        """
        Data: Graph G, point p ∈ P , candidate set V,
            distance threshold α ≥ 1, degree bound R
        Result: G is modified by setting at most R new
            out-neighbors for p
        """

        pass

    def index(self, L, R, repo: VectorDBRepository):
        """
        Data: Database P with n points where i-th point has coords xi, parameters α, L, R
        Result: Directed graph G over P with out-degree <=R
        """
        vector_ids = repo.get_vector_ids()
        self.graph = self.random_regular_graph(vector_ids, R)
        n = len(vector_ids)

        sigma = copy(vector_ids)
        shuffle(vector_ids)
        mediod = self.get_mediod(repo)
        if not mediod:
            raise Exception("couldn't find mediod")

        for i in range(n):
            query = repo.get_vector_by_id_lite(sigma[i])
            if not query:
                raise Exception("couldn't find query vec")
            (l, v) = self.greedy_search(mediod, query, 1, L, repo)  # type: ignore

    def search(self):
        pass

    def get_neighbors(
        self, v: VectorLite, repo: VectorDBRepository
    ) -> List[VectorLite]:
        neighbor_ids = self.graph.get(v.id)
        if not neighbor_ids:
            return []
        return repo.get_vectors_by_ids_lite(neighbor_ids)

    def get_mediod(self, repo: VectorDBRepository) -> Optional[VectorLite]:
        # TODO: Replace this function with an optimized vectorized version
        sample = repo.get_random_sample(config.INDEX_RND_SAMPLE_SIZE)
        mediod = None
        minimum = float("inf")
        for x in sample:
            for y in sample:
                if x.id == y.id:
                    continue
                d = self.distance(x, y)
                if d < minimum:
                    minimum = d
                    mediod = x
        return mediod

    @staticmethod
    def random_regular_graph(ids: List[int], r: int) -> Dict[int, List[int]]:
        all_ids = set(ids)
        graph = {}

        for id in all_ids:
            neighbors = random.sample(list(all_ids - {id}), r)
            graph[id] = neighbors

        return graph

    @staticmethod
    def distance(x: VectorLite, y: VectorLite) -> float:
        return float(np.linalg.norm(x.numpy_vector - y.numpy_vector))

    def save_index(self, repo: VectorDBRepository):
        repo.save_graph(self.graph)
        # repo.add_index_metadata("entry_point", str(self.entry_point))

    def load_index(self, repo: VectorDBRepository):
        self.graph = repo.get_graph()
        