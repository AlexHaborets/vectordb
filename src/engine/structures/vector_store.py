import random
from typing import Dict, List, Optional

from src.common import config
import numpy as np

from src.schemas.vector import VectorData


class VectorStore:
    """
    An in-memory vector cache/store
    """

    def __init__(
            self, 
            dims: int, 
            vectors: Optional[List[VectorData]] = None
        ) -> None:
        STARTING_SIZE = 512
        self.dims = dims
        self._size = 0

        # Maps db ids to matrix ids
        self.dbid_to_idx: Dict[int, int] = {}
        self.idx_to_dbid: List[int] = []

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
    def size(self) -> int:
        return self._size

    def resize(self, new_size: int) -> None:
        curr_capacity = self.capacity

        if new_size <= curr_capacity:
            return

        GROWTH_FACTOR = 1.5

        new_capacity = max(new_size, int(GROWTH_FACTOR * curr_capacity))

        new_vectors = np.empty(
            shape=(new_capacity, self.dims), 
            dtype=config.NUMPY_DTYPE
        )

        curr_size = self._size
        new_vectors[:curr_size] = self.vectors[:curr_size]

        self.vectors = new_vectors

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
                modified_ids.append(idx)
            else:
                new_vectors.append(v.vector)
                new_db_ids.append(v.id)

        if new_vectors:
            curr_size = self._size
            new_size = curr_size + len(new_vectors)
            self.resize(new_size)
            
            new_block = np.array(new_vectors, dtype=config.NUMPY_DTYPE)
            self.vectors[curr_size:new_size] = new_block

            self.idx_to_dbid.extend(new_db_ids)

            for i, db_id in enumerate(new_db_ids):
                idx = curr_size + i
                self.dbid_to_idx[db_id] = idx
                modified_ids.append(idx)
            
            self._size = new_size

        return modified_ids

    def get_random_sample(self, k: int) -> tuple[np.ndarray, List[int]]:
        """
        Returns k random vectors and a list of their ids
        """
        curr_size = self._size
        if curr_size > k:
            ids = random.sample(range(curr_size), k)
            return self.vectors[ids], ids
        else:
            return self.vectors[:curr_size], list(range(curr_size))
        
    def get_vectors(self) -> np.ndarray:
        return self.vectors[:self.size]

    def get_dbids(self) -> List[int]:
        return [i for i in self.dbid_to_idx.keys()]

    def get_idxs(self) -> np.ndarray:
        return np.arange(self.size)

    def get_dbid(self, idx: int) -> int:
        return self.idx_to_dbid[idx]

    def get_idx(self, dbid: int) -> int:
        return self.dbid_to_idx[dbid]
