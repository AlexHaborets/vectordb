import random
from typing import Dict, List

from src.common import config
from src.schemas import VectorLite
import numpy as np


class VectorStore:
    """
    An in-memory vector cache/store
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.vectors: np.ndarray = np.empty((0, dims), dtype=config.NUMPY_DTYPE)
        # Maps db ids to matrix ids
        self.dbid_to_idx: Dict[int, int] = {}
        # Maps matrix to ids db ids
        self.idx_to_dbid: Dict[int, int] = {}

        # Indeces in the vectors field of deleted vectors 
        # TODO: update methods to use this field
        self.idxs: set[int]

    def get(self, vector_id: int) -> np.ndarray:
        idx = self.dbid_to_idx[vector_id]
        return self.vectors[idx]

    def get_batch(self, vector_ids: List[int]) -> np.ndarray:
        if not vector_ids:
            return np.empty((0, self.dims), dtype=config.NUMPY_DTYPE)

        indeces = [self.dbid_to_idx[vid] for vid in vector_ids]
        return self.vectors[indeces]

    def add(self, vector: VectorLite) -> None:
        idx = self.vectors.shape[0]
        self.vectors = np.vstack([self.vectors, vector.numpy_vector])

        self.dbid_to_idx[vector.id] = idx
        self.idx_to_dbid[idx] = vector.id

    def set(self, vector: VectorLite) -> None:
        if vector.id in self.dbid_to_idx:
            idx = self.dbid_to_idx[vector.id]
            self.vectors[idx] = vector.numpy_vector
        else:
            self.add(vector)

    def add_batch(self, vectors: List[VectorLite]) -> None:
        new_vectors = [v for v in vectors if v.id not in self.dbid_to_idx]
        if not new_vectors:
            return

        new_db_ids = [v.id for v in new_vectors]
        new_np_vectors = np.array(
            [v.numpy_vector for v in new_vectors], dtype=config.NUMPY_DTYPE
        )

        current_size = self.vectors.shape[0]

        self.vectors = np.vstack([self.vectors, new_np_vectors])

        for i, db_id in enumerate(new_db_ids):
            new_idx = current_size + i
            self.dbid_to_idx[db_id] = new_idx
            self.idx_to_dbid[new_idx] = db_id

    def get_random_sample(self, k: int) -> Dict[int, np.ndarray]:
        db_ids = list(self.dbid_to_idx.keys())
        if len(db_ids) > k:
            db_ids = random.sample(db_ids, k)
        return {db_id: self.vectors[self.dbid_to_idx[db_id]] for db_id in db_ids}

    @classmethod
    def build_from_vectors(cls, vectors: List[VectorLite], dims: int) -> "VectorStore":
        store = cls(dims)

        if vectors:
            store.add_batch(vectors)

        return store

    def get_dbids(self) -> List[int]:
        return [i for i in self.dbid_to_idx.keys()]
