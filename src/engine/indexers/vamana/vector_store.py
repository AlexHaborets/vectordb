import random
from typing import Dict, List, Optional

import numpy as np

from src.common import config
from src.engine.indexers.vamana.ops import (
    clear_bit,
    create_bitset,
    resize_bitset,
    set_bit,
)
from src.schemas.vector import VectorData


class VectorStore:
    """
    An in-memory vector cache/store
    """

    def __init__(self, dims: int, vectors: Optional[List[VectorData]] = None) -> None:
        STARTING_SIZE = 512
        self.dims = dims
        self._size = 0

        # Maps db ids to matrix ids
        self.dbid_to_idx: Dict[int, int] = {}
        self.idx_to_dbid: List[int] = []

        deleted_size = max(STARTING_SIZE, len(vectors) if vectors else 0)
        self._deleted = create_bitset(deleted_size)
        self._free_idxs = []

        if vectors and len(vectors) > 0:
            self.vectors = np.array(
                [v.vector for v in vectors], dtype=config.NUMPY_DTYPE
            )
            self._size = self.vectors.shape[0]

            self.idx_to_dbid = [v.id for v in vectors]

            for i, v in enumerate(vectors):
                self.dbid_to_idx[v.id] = i
        else:
            self.vectors = np.empty((STARTING_SIZE, dims), dtype=config.NUMPY_DTYPE)

    @property
    def capacity(self) -> int:
        return self.vectors.shape[0]

    @property
    def total_size(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size - len(self._free_idxs)

    @property
    def deleted(self) -> np.ndarray:
        return self._deleted

    def resize(self, new_size: int) -> None:
        curr_capacity = self.capacity

        if new_size <= curr_capacity:
            return

        GROWTH_FACTOR = 1.5

        new_capacity = max(new_size, int(GROWTH_FACTOR * curr_capacity))
        new_vectors = np.empty(
            shape=(new_capacity, self.dims), dtype=config.NUMPY_DTYPE
        )
        new_vectors[: self._size] = self.vectors[: self._size]
        self.vectors = new_vectors

        self._deleted = resize_bitset(self._deleted, new_capacity)

    def get(self, idx: int) -> np.ndarray:
        return self.vectors[idx]

    def get_batch(self, idxs: List[int]) -> np.ndarray:
        if not idxs:
            return np.empty((0, self.dims), dtype=config.NUMPY_DTYPE)

        return self.vectors[idxs]

    def upsert_batch(self, vectors: List[VectorData]) -> List[int]:
        """
        Upserts vectors and returns list of modified ids
        """
        modified_ids: List[int] = []
        new_vectors: List[np.ndarray] = []
        new_db_ids: List[int] = []

        for v in vectors:
            if v.id in self.dbid_to_idx:
                idx = self.dbid_to_idx[v.id]
                self.vectors[idx] = v.vector
                clear_bit(self._deleted, idx)
                modified_ids.append(idx)
            else:
                new_vectors.append(v.vector)
                new_db_ids.append(v.id)

        if new_vectors:
            new_count = len(new_vectors)
            assigned_ids: List[int] = []

            while self._free_idxs and len(assigned_ids) < new_count:
                idx = self._free_idxs.pop()
                clear_bit(self._deleted, idx)
                assigned_ids.append(idx)

            reused_count = len(assigned_ids)
            appended_count = new_count - reused_count

            if appended_count > 0:
                new_size = self._size + appended_count
                self.resize(new_size)
                appended_ids = list(range(self._size, self._size + appended_count))
                assigned_ids.extend(appended_ids)
                self._size = new_size

            new_block = np.array(new_vectors, dtype=config.NUMPY_DTYPE)
            self.vectors[assigned_ids] = new_block

            # pad idx_to_dbid
            while len(self.idx_to_dbid) < self._size:
                self.idx_to_dbid.append(-1)

            for idx, db_id in zip(assigned_ids, new_db_ids):
                self.dbid_to_idx[db_id] = idx
                self.idx_to_dbid[idx] = db_id
                modified_ids.append(idx)

        return modified_ids

    def delete_batch(self, db_ids: List[int]):
        for db_id in db_ids:
            if db_id not in self.dbid_to_idx:
                continue

            idx = self.dbid_to_idx.pop(db_id)
            self._free_idxs.append(idx)
            set_bit(self._deleted, idx)
            self.idx_to_dbid[idx] = -1

    def get_random_sample(self, k: int) -> tuple[np.ndarray, List[int]]:
        """
        Returns k random vectors and a list of their ids
        """
        active_idxs = list(self.dbid_to_idx.values())
        if len(active_idxs) > k:
            sampled_idxs = random.sample(active_idxs, k)
        else:
            sampled_idxs = active_idxs

        return self.vectors[sampled_idxs], sampled_idxs

    def get_dbids(self) -> List[int]:
        return list(self.dbid_to_idx.keys())

    def get_idxs(self) -> np.ndarray:
        return np.array(list(self.dbid_to_idx.values()), dtype=np.int32)

    def get_dbid(self, idx: int) -> int:
        return self.idx_to_dbid[idx]

    def get_idx(self, dbid: int) -> int:
        return self.dbid_to_idx[dbid]
