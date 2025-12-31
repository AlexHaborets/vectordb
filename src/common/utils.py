from typing import List
import numpy as np

from src.common.config import NUMPY_DTYPE


def vector_to_bytes(vector: List[float]) -> bytes:
    return np.array(vector, dtype=NUMPY_DTYPE).tobytes()


def bytes_to_ndarray(buffer: bytes) -> np.ndarray:
    return np.frombuffer(buffer, dtype=NUMPY_DTYPE)


def bytes_to_vector(buffer: bytes) -> List[float]:
    return bytes_to_ndarray(buffer).tolist()
