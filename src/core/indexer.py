import heapq
from typing import Dict, List, Tuple
import config
import numpy as np


class Indexer:
    def __init__(self, dims: int):
        self.vectors: np.ndarray = np.empty((0, dims), dtype=np.float32)
        self.graph: Dict[int, List[int]] = {}
        self.entry_point: int

    def greedy_search(self, x_q, k) -> List[Tuple[float, int]]:
        entry_vec = self.vectors[self.entry_point]
        dist = self.distance(x_q, entry_vec)

        # Store a max heap for results
        results = [(-dist, self.entry_point)]
        # Store a min heap where the priority is the distance from x_q to vector
        candidates = [(dist, self.distance(x_q, entry_vec))]

        visited = {self.entry_point}

        while candidates:
            dist_p, p_id = heapq.heappop(candidates)

            dist_worst = -results[0][0]
            if dist_p > dist_worst and len(results) >= k:
                break

            for n_id in self.graph.get(p_id, set()):
                if n_id in visited:
                    continue

                visited.add(n_id)  # Mark visited
                n_vec = self.vectors[n_id]
                n_dist = self.distance(x_q, n_vec)
                if len(results) < k or n_dist < -results[0][0]:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results, (-n_dist, n_id))

                    if len(results) > k:
                        heapq.heappop(results)

        final_results = [(-d, i) for d, i in results]
        return final_results
    
    # def robust_prune(self, p, candidates):
    #     neighbors = []


    # def index(self):
    #     b = 1
    #     random_perm = np.random.permutation(n)
    #     for i in range(n):
    #         self.robust_prune()

    def search():
        pass

    @staticmethod
    def distance(x: np.array, y: np.array):
        return np.linalg.norm(x - y)
    
    def save_index(self, path: str):
        pass

    def load_index(self, path: str):
        pass

# Create a global instance of the Indexer
indexer = Indexer(dimensions=config.VECTOR_DIMENSIONS) 