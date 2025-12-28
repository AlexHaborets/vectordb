import random
from typing import Dict, List, Optional

from src.common import config
import numpy as np

from src.schemas.vector import VectorData


class VectorStore:
    """
    An in-memory vector cache/store
    """

    def __init__(self, dims: int, vectors: Optional[List[VectorData]] = None) -> None:
        self.dims = dims

        # Maps db ids to matrix ids
        self.dbid_to_idx: Dict[int, int] = {}
        self.idx_to_dbid: List[int] = []

        if vectors and len(vectors) > 0:
            self.vectors = np.array(
                [v.vector for v in vectors], dtype=config.NUMPY_DTYPE
            )

            self.idx_to_dbid = [v.id for v in vectors]

            for i, v in enumerate(vectors):
                self.dbid_to_idx[v.id] = i
        else:
            self.vectors = np.empty((0, dims), dtype=config.NUMPY_DTYPE)

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
        modified_ids = []
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
            current_size = self.size

            self.vectors = np.vstack([self.vectors, new_vectors])

            self.idx_to_dbid.extend(new_db_ids)

            for i, db_id in enumerate(new_db_ids):
                idx = current_size + i
                self.dbid_to_idx[db_id] = idx
                modified_ids.append(idx)

        return modified_ids

    def get_random_sample(self, k: int) -> tuple[np.ndarray, List[int]]:
        """
        Returns k random vectors and
        """
        current_size = self.vectors.shape[0]
        if current_size > k:
            ids = random.sample(range(current_size), k)
            return self.vectors[ids], ids
        else:
            return self.vectors, list(range(current_size))

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    def get_dbids(self) -> List[int]:
        return [i for i in self.dbid_to_idx.keys()]

    def get_idxs(self) -> List[int]:
        return list(range(self.size))

    def get_dbid(self, idx: int) -> int:
        return self.idx_to_dbid[idx]

    def get_idx(self, dbid: int) -> int:
        return self.dbid_to_idx[dbid]
