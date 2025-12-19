from typing import List

import numba as nb
import numpy as np

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


@nb.njit(fastmath=True, parallel=True, inline="always")
def compute_dists_batch(
    query: np.ndarray, targets: np.ndarray, metric: int
) -> np.ndarray:
    if metric == 0:
        return compute_dists_batch_l2(query, targets)
    else:
        return compute_dists_batch_cosine(query, targets)


@nb.njit(fastmath=True)
def insort(
    ids: np.ndarray,
    dists: np.ndarray,
    new_id: int,
    new_dist: float,
    curr_size: int,
    max_size: int,
) -> int:
    if curr_size >= max_size and new_dist >= dists[curr_size - 1]:
        return curr_size

    insert_ptr = 0
    while insert_ptr < curr_size:
        if new_dist < dists[insert_ptr]:
            break
        insert_ptr += 1

    if insert_ptr < curr_size:
        end = curr_size
        if curr_size >= max_size:
            end = max_size - 1

        for i in range(end, insert_ptr, -1):
            ids[i] = ids[i - 1]
            dists[i] = dists[i - 1]

    if insert_ptr < max_size:
        ids[insert_ptr] = new_id
        dists[insert_ptr] = new_dist
        if curr_size < max_size:
            return curr_size + 1

    return curr_size


@nb.njit(fastmath=True)
def greedy_search(
    entry_id: int,
    query_vector: np.ndarray,
    k: int,
    L: int,
    graph: np.ndarray,
    vectors: np.ndarray,
    seen: np.ndarray,
    metric: int,
) -> tuple[np.ndarray, np.ndarray, List]:
    """
    Data: Graph G with start node s, query xq, result
        size k, search list size L ≥ k
    Result: Result set L containing k-approx NNs, and
        a set V containing all the visited nodes
    """
    query_entry_dist = compute_dist(a=query_vector, b=vectors[entry_id], metric=metric)

    # create a candidate priority queue with two ndarrays
    candidates_dists = np.full(shape=L, fill_value=np.inf, dtype=NUMPY_DTYPE)
    candidates_ids = np.full(shape=L, fill_value=-1, dtype=np.int32)

    # add entry to the candidates
    candidates_dists[0] = query_entry_dist
    candidates_ids[0] = entry_id
    # number of candidates in the queue
    candidate_count = 1

    # mark entry as seen
    seen[entry_id] = True

    visited: List[int] = []

    # pointer to the candidate we haven't visited yet (currently entry point)
    # also represents the number of nodes we have processed
    candidate_ptr = 0

    # candidate_ptr < candidate_count means
    # there are still candidates to visit
    while candidate_ptr < candidate_count:
        if candidate_ptr >= L:
            break

        # pop a node from the candidate queue
        pstar_id = candidates_ids[candidate_ptr]
        candidate_ptr += 1

        visited.append(pstar_id)

        neighbors = graph[pstar_id]

        for n_id in neighbors:
            if n_id == -1:
                break
            if not seen[n_id]:
                seen[n_id] = True

                dist = compute_dist(a=vectors[n_id], b=query_vector, metric=metric)

                candidate_count = insort(
                    ids=candidates_ids,
                    dists=candidates_dists,
                    new_id=n_id,
                    new_dist=dist,
                    curr_size=candidate_count,
                    max_size=L,
                )

    return candidates_ids[:k], candidates_dists[:k], visited

