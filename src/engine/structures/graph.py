from typing import Dict, List

import numpy as np


class Graph:
    def __init__(self, size: int, degree: int) -> None:
        self._R = degree
        self._size = size
        self.graph: np.ndarray = np.full(
            fill_value=-1, shape=(size, degree), dtype=np.int32
        )

    @classmethod
    def from_db(
        cls, db_graph: Dict[int, List[int]], dbid_to_idx: Dict[int, int], degree: int
    ) -> "Graph":
        size = len(dbid_to_idx)
        graph = cls(size, degree)

        for dbid, db_neigbors in db_graph.items():
            if dbid not in dbid_to_idx:
                continue

            idx = dbid_to_idx[dbid]
            neighbors = {dbid_to_idx[n] for n in db_neigbors if n in dbid_to_idx}
            graph.set_neighbors(idx, neighbors)

        return graph

    @classmethod
    def random_regular(cls, size: int, degree: int) -> "Graph":
        graph = cls(size, degree)

        if size <= degree + 1:
            for idx in range(size):
                neighbors = {n for n in range(size) if n != idx}
                graph.set_neighbors(idx, neighbors)
            return graph

        for idx in range(size):
            candidates = np.random.choice(a=size, size=degree + 1, replace=False)
            candidates = candidates[candidates != idx]
            graph.graph[idx] = candidates[:degree]

        return graph

    @property
    def capacity(self) -> int:
        return self.graph.shape[0]

    def resize(self, new_size: int) -> None:
        curr_capacity = self.capacity

        if new_size <= curr_capacity:
            self._size = new_size
            return

        GROWTH_FACTOR = 1.5

        new_capacity = max(new_size, int(GROWTH_FACTOR * curr_capacity))

        new_graph = np.full(
            fill_value=-1, shape=(new_capacity, self._R), dtype=np.int32
        )

        new_graph[:curr_capacity, :] = self.graph

        self.graph = new_graph
        self._size = new_size

    def set_neighbors(self, idx: int, neighbors: set[int]) -> None:
        neighbors_list = list(neighbors)
        count = len(neighbors_list)

        if count > self._R:
            neighbors_list = neighbors_list[: self._R]
            count = self._R

        self.graph[idx, :count] = neighbors_list

        if count < self._R:
            self.graph[idx, count:] = -1

    def get_neighbors(self, idx: int) -> List[int]:
        row = self.graph[idx]
        return row[row != -1].tolist()

    def to_db_graph(self, idx_to_dbid: List[int]) -> Dict[int, List[int]]:
        graph = {}

        for i in range(self._size):
            row = self.graph[i]
            node_db_id = idx_to_dbid[i]
            neighbor_db_ids = [idx_to_dbid[n] for n in row if n != -1]
            graph[node_db_id] = neighbor_db_ids

        return graph
    
    def get_subgraph(self, ids: List[int], idx_to_dbid: List[int]) -> Dict[int, List[int]]:
        graph = {}

        for i in ids:
            row = self.graph[i]
            node_db_id = idx_to_dbid[i]
            neighbor_db_ids = [idx_to_dbid[n] for n in row if n != -1]
            graph[node_db_id] = neighbor_db_ids

        return graph
