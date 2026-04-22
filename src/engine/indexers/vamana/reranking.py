import numba as nb
import numpy as np

from src.common.config import NUMPY_DTYPE
from src.engine.indexers.vamana.ops import NUMBA_OPTIONS, compute_dist


@nb.njit(inline="always", **NUMBA_OPTIONS)
def dist_to_sim(dist: float, metric: int) -> float:
    if metric == 0:
        return 1.0 / (1.0 + dist)
    else:
        return 1.0 - dist


@nb.njit(inline="always", **NUMBA_OPTIONS)
def dists_to_sims(dists: np.ndarray, metric: int) -> np.ndarray:
    n = dists.shape[0]
    sims = np.empty(n, dtype=NUMPY_DTYPE)
    for i in range(n):
        sims[i] = dist_to_sim(dists[i], metric)
    return sims


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_sim(a: np.ndarray, b: np.ndarray, metric: int) -> float:
    dist = compute_dist(a, b, metric)
    return dist_to_sim(dist, metric)


@nb.njit(**NUMBA_OPTIONS)
def mmr_rerank(
    query_dists: np.ndarray,
    candidate_ids: np.ndarray,
    vectors: np.ndarray,
    metric: int,
    k: int,
    mmr_lambda: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    n = candidate_ids.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=NUMPY_DTYPE)

    k = min(k, n)

    query_sims = dists_to_sims(query_dists, metric)

    selected_ids = np.empty(k, dtype=np.int32)
    selected_sims = np.empty(k, dtype=NUMPY_DTYPE)
    chosen = np.zeros(n, dtype=np.bool_)
    # map document -> the maximum similiarity of its similiarities to selected docs
    max_sim_to_selected = np.full(n, fill_value=-np.inf, dtype=NUMPY_DTYPE)

    # find the most similiar/relevant document
    best_id = int(np.argmax(query_sims))
    # mark it both as selected and chosen
    selected_ids[0] = candidate_ids[best_id]
    selected_sims[0] = query_sims[best_id]
    chosen[best_id] = True

    # iterate over the next k-1 documents to be selected
    for i in range(1, k):
        # last vector to be added to the selected set
        last_vec = vectors[candidate_ids[best_id]]

        local_best_id = -1
        local_best_score = -np.inf

        # iterate over other vectors that haven't been chosen yet
        # to find a vector that has the highest MMR score to the last added vector
        for j in range(n):
            if chosen[j]:
                continue

            new_vec = vectors[candidate_ids[j]]
            new_sim = compute_sim(new_vec, last_vec, metric)

            if new_sim > max_sim_to_selected[j]:
                max_sim_to_selected[j] = new_sim

            # calculate score using MMR formula/equation
            score = (mmr_lambda * query_sims[j]) - (
                (1.0 - mmr_lambda) * max_sim_to_selected[j]
            )

            if score > local_best_score:
                local_best_score = score
                local_best_id = j

        # update the best_id for the next iteration
        best_id = local_best_id
        # add to the selected set
        selected_ids[i] = candidate_ids[best_id]
        selected_sims[i] = query_sims[best_id]
        chosen[best_id] = True

    return (selected_ids, selected_sims)
