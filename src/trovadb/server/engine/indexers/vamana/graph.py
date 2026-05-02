from typing import Dict, List, Optional

import numpy as np

from trovadb.server.engine.snapshot import IndexSnapshot


class Graph:
    def __init__(self, size: int, degree: int) -> None:
        self._R = degree
        self._size = size
        self.graph: np.ndarray = np.full(
            fill_value=-1, shape=(size, degree), dtype=np.int32
        )

    def to_snapshot(
        self, idx_to_dbid: List[int], entry_point: int
    ) -> Optional[IndexSnapshot]:
        """
        Compacts the graph by removing deleted nodes
        """
        dbids = np.array(idx_to_dbid[: self._size], dtype=np.int64)
        active_ids = np.where(dbids != -1)[0].astype(np.int32)
        active_count = active_ids.shape[0]
        remap = np.full(self._size, -1, dtype=np.int32)

        if active_count == 0:
            return None

        remap[active_ids] = np.arange(active_count, dtype=np.int32)
        compacted_graph = self.graph[active_ids].copy()
        valid = compacted_graph != -1
        compacted_graph[valid] = remap[compacted_graph[valid]]
        compacted_id_map = dbids[active_ids]

        return IndexSnapshot(
            graph=compacted_graph,
            id_map=compacted_id_map,
            entry_point=int(remap[entry_point]),
        )

    @classmethod
    def from_snapshot(cls, snapshot: np.ndarray, degree: int) -> "Graph":
        size = snapshot.shape[0]
        graph = cls(size, degree)
        graph.graph[:size] = snapshot
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

        curr_size = self._size
        new_graph[:curr_size] = self.graph[:curr_size]

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

    def get_subgraph(
        self, ids: List[int], idx_to_dbid: List[int]
    ) -> Dict[int, List[int]]:
        graph = {}

        for i in ids:
            row = self.graph[i]
            node_db_id = idx_to_dbid[i]
            if node_db_id == -1:
                continue
            neighbor_db_ids = [idx_to_dbid[n] for n in row if n != -1]
            graph[node_db_id] = neighbor_db_ids

        return graph
