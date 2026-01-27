import pytest
import numpy as np
from dataset_loader import load_dataset

@pytest.mark.parametrize("collection", [{"dim": 128, "name": "sift_small", "distance": "l2"}], indirect=True)
def test_recall_siftsmall(collection):
    print("\n[Init] Loading SIFT-Small...")
    with load_dataset("siftsmall") as (base, query, ground_truth):
        benchmark_recall(collection, base, query, ground_truth, expected_recall=0.90)

# @pytest.mark.parametrize("collection", [{"dim": 128, "name": "sift", "distance": "l2"}], indirect=True)
# def test_recall_sift(collection):
#     print("\n[Init] Loading SIFT...")

#     with load_dataset("sift") as (base, query, ground_truth):
#         benchmark_recall(collection, base, query, ground_truth, expected_recall=0.90)

def benchmark_recall(collection, base_vecs, query_vecs, ground_truth, expected_recall) -> None:
    """
    Benchmarks recall on a given dataset
    """
    
    N_VEC = len(base_vecs) 
    K_SEARCH = 10
    BATCH_SIZE = 5000 

    print(f"[Write] Upserting {N_VEC} vectors...")
    
    for i in range(0, N_VEC, BATCH_SIZE):
        limit = min(i + BATCH_SIZE, N_VEC)
        
        batch_ids = [str(j) for j in range(i, limit)]
        
        batch_vecs = base_vecs[i : limit].tolist()
        
        collection.upsert(
            ids=batch_ids, 
            vectors=batch_vecs
        )

        pct = (limit / N_VEC) * 100
        print(f"\r[Write] Progress: {limit}/{N_VEC} ({pct:.1f}%)", end="", flush=True)

    print("") 
    print(f"[Read] Running {len(query_vecs)} queries...")
    
    hits = 0
    total_expected = len(query_vecs) * K_SEARCH
    
    query_vecs_loaded = query_vecs[:] 
    
    for i, query_vec in enumerate(query_vecs_loaded):
        results = collection.search(query=query_vec.tolist(), k=K_SEARCH)
        
        retrieved_ids = [int(r.vector.id) for r in results]
        true_ids = ground_truth[i][:K_SEARCH]
        
        hits += len(np.intersect1d(retrieved_ids, true_ids))

    recall = hits / total_expected
    print(f"\n[Result] Recall@{K_SEARCH}: {recall:.4f}")
    
    assert recall >= expected_recall, f"Recall {recall:.4f} < {expected_recall}"