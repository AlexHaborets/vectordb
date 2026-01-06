import numba as nb
import numpy as np

from src.common.config import NUMPY_DTYPE


@nb.njit(fastmath=True, inline="always", cache=True)
def compute_dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    sum_value = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        sum_value += diff * diff
    return np.sqrt(sum_value)


@nb.njit(fastmath=True, inline="always", cache=True)
def compute_dist_cosine(a: np.ndarray, b: np.ndarray) -> float:
    sum_value = 0.0
    for i in range(a.shape[0]):
        sum_value += a[i] * b[i]
    return 1 - sum_value


@nb.njit(fastmath=True, inline="always", cache=True)
def compute_dist(a: np.ndarray, b: np.ndarray, metric: int) -> float:
    if metric == 0:
        return compute_dist_l2(a, b)
    else:
        return compute_dist_cosine(a, b)


@nb.njit(fastmath=True, inline="always", cache=True)
def compute_dists_batch_l2(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = targets.shape[0]
    dists = np.empty(n, dtype=NUMPY_DTYPE)

    for i in nb.prange(n):
        dists[i] = compute_dist_l2(a=query, b=targets[i])

    return dists


@nb.njit(fastmath=True, inline="always", cache=True)
def compute_dists_batch_cosine(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = targets.shape[0]
    dists = np.empty(n, dtype=NUMPY_DTYPE)

    for i in nb.prange(n):
        dists[i] = compute_dist_cosine(a=query, b=targets[i])

    return dists


@nb.njit(fastmath=True, inline="always", cache=True)
def compute_dists_batch(
    query: np.ndarray, targets: np.ndarray, metric: int
) -> np.ndarray:
    if metric == 0:
        return compute_dists_batch_l2(query, targets)
    else:
        return compute_dists_batch_cosine(query, targets)


@nb.njit(fastmath=True, cache=True)
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


@nb.njit(fastmath=True, cache=True)
def count_neighbors(neighbors_array: np.ndarray) -> int:
    count = 0
    for i in range(neighbors_array.shape[0]):
        if neighbors_array[i] == -1:
            break
        count += 1
    return count


@nb.njit(fastmath=True, cache=True)
def greedy_search(
    entry_id: int,
    query_vector: np.ndarray,
    k: int,
    L: int,
    seen: np.ndarray,
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

    # add entry to the candidates
    candidates_dists[0] = query_entry_dist
    candidates_ids[0] = entry_id
    # number of candidates in the queue
    candidate_count = 1

    # mark entry as seen
    seen[entry_id] = True

    visited = np.empty(L*2, dtype=np.int32)
    visited_count = 0

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

        visited[visited_count] = pstar_id
        visited_count += 1

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

    # Reset the visited mask
    for i in range(visited_count):
        visited_id = visited[i]
        seen[visited_id] = False

    return candidates_ids[:k], candidates_dists[:k], visited[:visited_count]


@nb.njit(fastmath=True, cache=True)
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
    

    # We can neglect the race conditions that will happen
    # as they degrade the graph quality only by a small margin
    # and the speed up is worth it
    graph[source_id] = neighbors


@nb.njit(fastmath=True, parallel=True, nogil=True, cache=True)
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

    for i in nb.prange(batch_size):
        seen = np.zeros(vectors.shape[0], dtype=np.bool_)
        query_id = batch_ids[i]
        query_vector = vectors[query_id]
        _, _, V = greedy_search(
            entry_id=entry_point,
            query_vector=query_vector,
            k=1,
            L=L,
            seen=seen,
            graph=graph,
            vectors=vectors,
            metric=metric
        )

        robust_prune(
            source_id=query_id, 
            candidates_ids=V, 
            alpha=alpha,
            R=R,
            graph=graph,
            vectors=vectors,
            metric=metric
        )

@nb.njit(fastmath=True, parallel=True, nogil=True, cache=True)
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
    # Set of ids for which the graph was modified
    modified_ids = np.zeros(vectors.shape[0], np.bool_)
    
    for i in nb.prange(batch_size):
        query_id = batch_ids[i] 

        query_neighbors = graph[query_id]

        for other in query_neighbors:
            # Stop if we reached the neighbor padding
            if other == -1:
                break
            
            other_neighbors = graph[other]
            
            duplicate = False
            empty_slot = -1

            for j in range(R):
                neighbor_id = other_neighbors[j]
                if neighbor_id == query_id:
                    duplicate = True 
                    break

                # Add i to neighbors
                if neighbor_id == -1:
                    empty_slot = j
                    break
            
            if duplicate:
                continue
            
            # Means there are now more than R neighbors (no space left)
            if empty_slot == -1:
                robust_prune(
                    source_id=other, 
                    # In the algorithm the candidates is other_neighbors + i
                    # But robust prune already adds the neighbors so we can 
                    # just pass i by itself
                    candidates_ids=np.array([query_id], dtype=np.int32),   
                    alpha=alpha, 
                    R=R,
                    graph=graph,
                    vectors=vectors,
                    metric=metric
                )
                
            else:
                # other_neighbors is a ref so we are modifying the graph itself
                other_neighbors[empty_slot] = query_id

            # Add other to modified set
            modified_ids[other] = True

    return np.flatnonzero(modified_ids) 