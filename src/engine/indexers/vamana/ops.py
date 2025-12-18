import numpy as np
import numba as nb
from src.common.config import NUMPY_DTYPE


@nb.njit(fastmath=True)
def compute_dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    sum_value = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        sum_value += diff * diff
    return np.sqrt(sum_value)


@nb.njit(fastmath=True)
def compute_dist_cosine(a: np.ndarray, b: np.ndarray) -> float:
    sum_value = 0.0
    for i in range(a.shape[0]):
        sum_value += a[i] * b[i]
    return 1 - sum_value


@nb.njit(fastmath=True)
def compute_dist(a: np.ndarray, b: np.ndarray, metric: int) -> float:
    if metric == 0:
        return compute_dist_l2(a, b)
    else:
        return compute_dist_cosine(a, b)


@nb.njit(fastmath=True, parallel=True)
def compute_dists_batch_l2(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = targets.shape[0]
    dists = np.empty(n, dtype=NUMPY_DTYPE)

    for i in nb.prange(n):
        dists[i] = compute_dist_l2(a=query, b=targets[i])

    return dists


@nb.njit(fastmath=True, parallel=True)
def compute_dists_batch_cosine(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = targets.shape[0]
    dists = np.empty(n, dtype=NUMPY_DTYPE)

    for i in nb.prange(n):
        dists[i] = compute_dist_cosine(a=query, b=targets[i])

    return dists


@nb.njit(fastmath=True, parallel=True)
def compute_dists_batch(
    query: np.ndarray, targets: np.ndarray, metric: int
) -> np.ndarray:
    if metric == 0:
        return compute_dists_batch_l2(query, targets)
    else:
        return compute_dists_batch_cosine(query, targets)
