import numba as nb
import numpy as np

from src.common.config import NUMPY_DTYPE

NUMBA_OPTIONS = {
    "fastmath": True,
    "cache": True,
}


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    sum_value = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        sum_value += diff * diff
    return np.sqrt(sum_value)


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_dist_cosine(a: np.ndarray, b: np.ndarray) -> float:
    sum_value = 0.0
    for i in range(a.shape[0]):
        sum_value += a[i] * b[i]
    return 1 - sum_value


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_dist(a: np.ndarray, b: np.ndarray, metric: int) -> float:
    if metric == 0:
        return compute_dist_l2(a, b)
    else:
        return compute_dist_cosine(a, b)


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_dists_batch_l2(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = targets.shape[0]
    dists = np.empty(n, dtype=NUMPY_DTYPE)

    for i in nb.prange(n):
        dists[i] = compute_dist_l2(a=query, b=targets[i])

    return dists


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_dists_batch_cosine(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = targets.shape[0]
    dists = np.empty(n, dtype=NUMPY_DTYPE)

    for i in nb.prange(n):
        dists[i] = compute_dist_cosine(a=query, b=targets[i])

    return dists


@nb.njit(inline="always", **NUMBA_OPTIONS)
def compute_dists_batch(
    query: np.ndarray, targets: np.ndarray, metric: int
) -> np.ndarray:
    if metric == 0:
        return compute_dists_batch_l2(query, targets)
    else:
        return compute_dists_batch_cosine(query, targets)


@nb.njit(**NUMBA_OPTIONS)
def insort(
    ids: np.ndarray,
    dists: np.ndarray,
    checked: np.ndarray,
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
            checked[i] = checked[i - 1]

    if insert_ptr < max_size:
        ids[insert_ptr] = new_id
        dists[insert_ptr] = new_dist
        checked[insert_ptr] = False
        if curr_size < max_size:
            return curr_size + 1

    return curr_size


@nb.njit(**NUMBA_OPTIONS)
def count_neighbors(neighbors_array: np.ndarray) -> int:
    count = 0
    for i in range(neighbors_array.shape[0]):
        if neighbors_array[i] == -1:
            break
        count += 1
    return count


# Bitset operations
# in binary // 2^k is eq to >> k
# so x // 8 (which is x // 2^3):
# x // 8 is eq to x >> 3


# x % 8 is eq to x & 7
@nb.njit(**NUMBA_OPTIONS)
def set_bit(bitset: np.ndarray, index: int) -> None:
    bitset[index >> 3] |= 1 << (index & 7)


@nb.njit(**NUMBA_OPTIONS)
def get_bit(bitset: np.ndarray, index: int) -> bool:
    return (bitset[index >> 3] >> (index & 7)) & 1


@nb.njit(**NUMBA_OPTIONS)
def greedy_search(
    entry_id: int,
    query_vector: np.ndarray,
    k: int,
    L: int,
    seen: np.ndarray,  # bitset
    graph: np.ndarray,
    vectors: np.ndarray,
    metric: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    candidates_checked = np.zeros(shape=L, dtype=np.bool_)

    # add entry to the candidates
    candidates_dists[0] = query_entry_dist
    candidates_ids[0] = entry_id
    candidate_count = 1

    # mark entry as seen
    set_bit(seen, entry_id)

    # visited set
    visited = np.empty(L * 4, dtype=np.int32)
    visited_count = 0

    while True:
        candidate_ptr: int = -1

        for i in range(candidate_count):
            if not candidates_checked[i]:
                candidate_ptr = i
                break

        if candidate_ptr == -1:
            break

        # optimization
        if (
            candidate_count == L
            and candidates_dists[candidate_ptr] > candidates_dists[L - 1]
        ):
            break

        # pop a node from the candidate queue
        pstar_id = candidates_ids[candidate_ptr]
        candidates_checked[candidate_ptr] = True

        # avoid a buffer overflow for safety
        if visited_count < visited.shape[0]:
            visited[visited_count] = pstar_id
            visited_count += 1

        neighbors = graph[pstar_id]

        for n_id in neighbors:
            if n_id == -1:
                break

            if not get_bit(seen, n_id):
                set_bit(seen, n_id)

                dist = compute_dist(a=vectors[n_id], b=query_vector, metric=metric)

                candidate_count = insort(
                    ids=candidates_ids,
                    dists=candidates_dists,
                    checked=candidates_checked,
                    new_id=n_id,
                    new_dist=dist,
                    curr_size=candidate_count,
                    max_size=L,
                )

    return candidates_ids[:k], candidates_dists[:k], visited[:visited_count]


@nb.njit(**NUMBA_OPTIONS)
def robust_prune(
    source_id: int,
    candidates_ids: np.ndarray,
    alpha: float,
    R: int,
    graph: np.ndarray,
    vectors: np.ndarray,
    metric: int,
) -> None:
    """
    Data: Graph G, point p ∈ P , candidate set V,
        distance threshold α ≥ 1, degree bound R
    Result: G is modified by setting at most R new
        out-neighbors for p
    """
    # Create the candidate set V ← (V ∪ Nout(p))
    source_neighbors = graph[source_id]
    source_neighbors_count = count_neighbors(source_neighbors)

    merged_candidates = np.empty(
        shape=candidates_ids.shape[0] + source_neighbors_count, dtype=np.int32
    )
    merged_candidates[: candidates_ids.shape[0]] = candidates_ids
    merged_candidates[candidates_ids.shape[0] :] = source_neighbors[
        :source_neighbors_count
    ]

    candidates = np.unique(merged_candidates)

    candidate_count = candidates.shape[0]
    if candidate_count == 0:
        return

    source_vector = vectors[source_id]
    candidate_source_dists = compute_dists_batch(
        query=source_vector, targets=vectors[candidates], metric=metric
    )

    candidates_ids_sorted = np.argsort(candidate_source_dists)

    # initialize empty neighbors set
    neighbors = np.full(R, -1, dtype=np.int32)
    neighbor_count = 0

    for i in range(candidate_count):
        if neighbor_count >= R:
            break

        # equivalent of argmin in the algorithm
        argmin_id = candidates_ids_sorted[i]
        pstar_id = candidates[argmin_id]

        # equivalent of removing pstar from candidates in the algorithm
        if pstar_id == source_id:
            continue

        pstar_dist = candidate_source_dists[argmin_id]
        pstar_vec = vectors[pstar_id]

        keep = True

        for j in range(neighbor_count):
            neighbor_id = neighbors[j]
            neighbor_vec = vectors[neighbor_id]

            pstar_neighbor_dist = compute_dist(
                a=pstar_vec, b=neighbor_vec, metric=metric
            )

            if alpha * pstar_neighbor_dist <= pstar_dist:
                keep = False
                break

        if keep:
            neighbors[neighbor_count] = pstar_id
            neighbor_count += 1

    graph[source_id] = neighbors


@nb.njit(parallel=True, **NUMBA_OPTIONS)
def forward_indexing_pass(
    batch_ids: np.ndarray,
    alpha: float,
    R: int,
    L: int,
    entry_point: int,
    graph: np.ndarray,
    vectors: np.ndarray,
    metric: int,
) -> None:
    """
    Exploring the neighbors of the batch nodes
    """

    batch_size = batch_ids.shape[0]
    vector_count = vectors.shape[0]

    # (x + y - 1) // y
    # (x + 8 - 1) // 8
    bitset_size = (vector_count + 7) // 8

    for i in nb.prange(batch_size):
        query_id = batch_ids[i]
        query_vector = vectors[query_id]

        seen = np.zeros(bitset_size, dtype=np.uint8)

        _, _, visited = greedy_search(
            entry_id=entry_point,
            query_vector=query_vector,
            k=1,
            L=L,
            seen=seen,
            graph=graph,
            vectors=vectors,
            metric=metric,
        )

        robust_prune(
            source_id=query_id,
            candidates_ids=visited,
            alpha=alpha,
            R=R,
            graph=graph,
            vectors=vectors,
            metric=metric,
        )


@nb.njit(parallel=True, **NUMBA_OPTIONS)
def backward_indexing_pass(
    batch_ids: np.ndarray,
    alpha: float,
    R: int,
    graph: np.ndarray,
    vectors: np.ndarray,
    metric: int,
) -> np.ndarray:
    """
    Returns list of vector ids for which the graph was modified
    """
    batch_size = batch_ids.shape[0]

    # Robust prune modifies the source node in graph
    sources = np.empty(batch_size * R, dtype=np.int32)
    candidates = np.empty(batch_size * R, dtype=np.int32)
    edge_count = 0

    # To parallelize safely we need to make sure that robust prune
    # is only called for different sources at the same time to avoid race conditions
    for i in range(batch_size):
        query_id = batch_ids[i]
        query_neighbors = graph[query_id]

        for other_id in query_neighbors:
            # Stop if we reached the neighbor padding
            if other_id == -1:
                break

            candidates[edge_count] = query_id
            sources[edge_count] = other_id
            edge_count += 1

    if edge_count == 0:
        return np.empty(0, dtype=np.int32)

    # Now we need to group the candidates by sources

    sources_valid = sources[:edge_count]
    candidates_valid = candidates[:edge_count]

    #  get ids to sort sources and candidates
    sort_ids = np.argsort(sources_valid)

    sources_sorted = sources_valid[sort_ids]
    candidates_sorted = candidates_valid[sort_ids]

    # basically looks like this:
    # sources_sorted =    [1 1 1 2 2 2 2 2 3 3 3 3 3 ... ]
    # candidates_sorted = [4 1 2 4 6 3 2 7 5 2 5 8 4 ...]
    #                      ^     ^         ^         ^
    # segments =          [0     3         8         13 ... ]
    segments = np.empty(edge_count + 1, dtype=np.int32)
    segments[0] = 0
    unique_count = 1

    for i in range(1, edge_count):
        if sources_sorted[i] != sources_sorted[i - 1]:
            segments[unique_count] = i
            unique_count += 1

    segments[unique_count] = edge_count

    # Set of ids for which the graph was modified
    modified_map = np.zeros(unique_count, dtype=np.bool_)

    for i in nb.prange(unique_count):
        start = segments[i]
        end = segments[i + 1]

        source = int(sources_sorted[start])
        local_candidates = candidates_sorted[start:end]

        source_neighbors = graph[source]

        prune_candidates = np.empty(end - start, dtype=np.int32)
        prune_candidate_count = 0

        for candidate in local_candidates:
            duplicate = False
            empty_slot = -1

            # Check if candidate is already a neighbor for source
            for j in range(R):
                source_neighbor = source_neighbors[j]
                if source_neighbor == -1:
                    empty_slot = j
                    break
                if source_neighbor == candidate:
                    duplicate = True
                    break

            # Skip if candidate is already a neighbor for source
            if duplicate:
                continue

            if empty_slot == -1:
                prune_candidates[prune_candidate_count] = candidate
                prune_candidate_count += 1
            else:
                source_neighbors[empty_slot] = candidate

            modified_map[i] = True

        if prune_candidate_count > 0:
            robust_prune(
                source_id=source,
                candidates_ids=prune_candidates[:prune_candidate_count],
                alpha=alpha,
                R=R,
                graph=graph,
                vectors=vectors,
                metric=metric,
            )

    modified_count = 0
    for i in range(unique_count):
        if modified_map[i]:
            modified_count += 1

    modified = np.empty(modified_count, dtype=np.int32)
    modified_count = 0

    for i in range(unique_count):
        if modified_map[i]:
            mod_id = sources_sorted[segments[i]]
            modified[modified_count] = mod_id
            modified_count += 1

    return modified
