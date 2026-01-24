import pytest
import numpy as np
from dataset_loader import load_dataset

@pytest.mark.parametrize("collection", [{"dim": 128, "name": "sift_small", "distance": "l2"}], indirect=True)
def test_recall_siftsmall(collection):
    print("\n[Init] Loading SIFT-Small...")
    base, query, ground_truth = load_dataset("siftsmall")
    
    benchmark_recall(collection, base, query, ground_truth, expected_recall=0.90)

def benchmark_recall(collection, base_vecs, query_vecs, ground_truth, expected_recall) -> None:
    """
    Benchmarks recall on a given dataset

    Definition 1.1 (𝑘-recall@𝑘). For a query vector 𝑞 over
    dataset 𝑃, suppose that (a) 𝐺 ⊆ 𝑃 is the set of actual 𝑘 nearest
    neighbors in 𝑃, and (b) 𝑋 ⊆ 𝑃 is the output of a 𝑘-ANNS
    query to an index. Then the 𝑘-recall@𝑘 for the index for
    query 𝑞 is |𝑋 ∩𝐺 |
    𝑘 . Recall for a set of queries refers to the
    average recall over all queries.
    """

    N_VEC = len(base_vecs)
    K_SEARCH = 10
    BATCH_SIZE = 1000

    print(f"[Write] Upserting {N_VEC} vectors...")
    ids = [str(i) for i in range(N_VEC)]
    
    for i in range(0, N_VEC, BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_vecs = base_vecs[i : i + BATCH_SIZE].tolist()
        collection.upsert(
            ids=batch_ids, 
            vectors=batch_vecs,
            batch_size=512
        )

    print(f"[Read] Running {len(query_vecs)} queries...")
    
    hits = 0
    total_expected = len(query_vecs) * K_SEARCH
    
    for i, query_vec in enumerate(query_vecs):
        results = collection.search(query=query_vec.tolist(), k=K_SEARCH)
        
        retrieved_ids = [int(r.vector.id) for r in results]
        true_ids = ground_truth[i][:K_SEARCH]
        
        hits += len(np.intersect1d(retrieved_ids, true_ids))

    recall = hits / total_expected
    print(f"\n[Result] Recall@{K_SEARCH}: {recall:.4f}")
    
    assert recall >= expected_recall, f"Recall {recall:.4f} < {expected_recall}"