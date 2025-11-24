import random
from typing import Dict, List


class Graph:
    def __init__(self, graph: Dict[int, List[int]] = {}) -> None:
        self.graph = graph

    @classmethod
    def random_regular(cls, verteces: List[int], r: int) -> "Graph":
        all_verteces = set(verteces)
        graph = {}

        if len(all_verteces) <= r:
            for current_id in all_verteces:
                graph[current_id] = list(all_verteces - {current_id})
        else:
            for id in all_verteces:
                neighbors = random.sample(list(all_verteces - {id}), r)
                graph[id] = neighbors

        return Graph(graph)
    
    def set_neighbors(self, vector_id: int, neighbors: set[int]) -> None:
        self.graph[vector_id] = list(neighbors)

    def add_neighbors(self, vector_id: int, neighbors: set[int]) -> None:
        curr_neighbors = self.graph.get(vector_id, [])
        updated_neighbors = set(curr_neighbors) | neighbors
        self.graph[vector_id] = list(updated_neighbors)

    def get_neighbors(self, vector_id: int) -> List[int]:
        return self.graph.get(vector_id, [])    