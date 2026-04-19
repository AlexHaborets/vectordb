import csv
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest
from dataset_loader import load_dataset

# The file where the benchmark data will be stored
LOG_FILE = "benchmark_results.csv"


@dataclass
class BenchmarkThresholds:
    min_throughput: float  # vectors / second
    max_avg_latency: float  # milliseconds
    min_recall: float  # percentage (0.0 to 1.0)


def log_results_to_csv(
    experiment_id: str,
    dataset_name: str,
    l_search: int,
    duration: float,
    throughput: float,
    avg_latency: float,
    p95_latency: float,
    recall: float,
):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "experiment_id",
                    "timestamp",
                    "dataset",
                    "l_search",
                    "indexing_time_s",
                    "throughput_vps",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "recall_at_10",
                ]
            )

        writer.writerow(
            [
                experiment_id,
                datetime.now().isoformat(timespec="seconds"),
                dataset_name,
                l_search,
                f"{duration:.4f}",
                f"{throughput:.2f}",
                f"{avg_latency:.4f}",
                f"{p95_latency:.4f}",
                f"{recall:.4f}",
            ]
        )


def run_benchmark(
    collection, base_vecs, query_vecs, ground_truth, thresholds: BenchmarkThresholds
):
    N_VEC = len(base_vecs)
    N_QUERIES = len(query_vecs)
    K_SEARCH = 10
    L_SEARCH_VALUES = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    BATCH_SIZE = 5000
    WARMUP_QUERIES = 10

    experiment_id = str(uuid.uuid4())[:8]

    # ==========================================
    # 1. WRITE PHASE (Throughput & Indexing)
    # ==========================================
    print(f"\n[Write] Upserting {N_VEC} vectors (Experiment: {experiment_id})...")

    start_time = time.perf_counter()

    for i in range(0, N_VEC, BATCH_SIZE):
        limit = min(i + BATCH_SIZE, N_VEC)
        batch_ids = [str(j) for j in range(i, limit)]
        batch_vecs = base_vecs[i:limit].tolist()

        collection.upsert(ids=batch_ids, vectors=batch_vecs, batch_size=512)
        print(
            f"\r[Write] Progress: {limit}/{N_VEC} ({(limit / N_VEC) * 100:.1f}%)",
            end="",
            flush=True,
        )

    indexing_duration = time.perf_counter() - start_time
    throughput = N_VEC / indexing_duration

    # ==========================================
    # 2. READ PHASE (Pareto Sweep)
    # ==========================================
    dataset_name = collection.name

    print(f"\n\n{'=' * 85}")
    print(
        f" Pareto Sweep Results: {dataset_name.upper()} | Experiment ID: {experiment_id}"
    )
    print(
        f" Indexing Time: {indexing_duration:.2f} s | Throughput: {throughput:.2f} v/s"
    )
    print(f"{'-' * 85}")
    print(
        f" {'L_search':<10} | {'Recall@10':<15} | {'Avg Latency (ms)':<20} | {'P95 Latency (ms)':<20}"
    )
    print(f"{'-' * 85}")

    query_vecs_loaded = query_vecs[:]
    total_expected = N_QUERIES * K_SEARCH

    max_recall_achieved = 0.0
    latency_at_max_recall = float("inf")

    for l_search in L_SEARCH_VALUES:
        latencies = []
        hits = 0

        for i, query_vec in enumerate(query_vecs_loaded):
            # Time the search
            t0 = time.perf_counter()
            results = collection.search(
                query=query_vec.tolist(), k=K_SEARCH, L_search=l_search
            )
            t1 = time.perf_counter()

            if i >= WARMUP_QUERIES:
                latencies.append((t1 - t0) * 1000)  # ms

            # Calculate accuracy
            retrieved_ids = [int(r.vector.id) for r in results]
            true_ids = ground_truth[i][:K_SEARCH]
            hits += len(np.intersect1d(retrieved_ids, true_ids))

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        recall = hits / total_expected

        if recall > max_recall_achieved:
            max_recall_achieved = recall
            latency_at_max_recall = avg_latency

        print(
            f" {l_search:<10} | {recall:<15.4f} | {avg_latency:<20.2f} | {p95_latency:<20.2f}"
        )

        log_results_to_csv(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            l_search=l_search,
            duration=indexing_duration,
            throughput=throughput,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            recall=recall,
        )

    print(f"{'=' * 85}\n")

    # ==========================================
    # 3. ASSERTIONS
    # ==========================================
    assert throughput >= thresholds.min_throughput, (
        f"Write too slow! Got {throughput:.2f} v/s, need {thresholds.min_throughput} v/s"
    )

    assert max_recall_achieved >= thresholds.min_recall, (
        f"Recall too low! Max achieved {max_recall_achieved:.4f}, need {thresholds.min_recall}"
    )

    assert latency_at_max_recall <= thresholds.max_avg_latency, (
        f"Read too slow at max recall! Avg: {latency_at_max_recall:.2f}ms, Limit: {thresholds.max_avg_latency}ms"
    )


# ==========================================
# PYTEST RUNNERS
# ==========================================


# @pytest.mark.parametrize(
#     "collection, dataset_name, thresholds",
#     [
#         (
#             {"dim": 128, "name": "sift_small", "distance": "l2"},
#             "siftsmall",
#             BenchmarkThresholds(
#                 min_throughput=700.0, max_avg_latency=15.0, min_recall=0.90
#             ),
#         )
#     ],
#     indirect=["collection"],
# )
# def test_benchmark_siftsmall(collection, dataset_name, thresholds):
#     with load_dataset(dataset_name) as (base, query, ground_truth):
#         run_benchmark(collection, base, query, ground_truth, thresholds)


# @pytest.mark.parametrize(
#     "collection, dataset_name, thresholds",
#     [
#         (
#             {"dim": 128, "name": "sift", "distance": "l2"},
#             "sift",
#             BenchmarkThresholds(min_throughput=1500.0, max_avg_latency=20.0, min_recall=0.90)
#         )
#     ],
#     indirect=["collection"]
# )
# def test_benchmark_sift(collection, dataset_name, thresholds):
#     with load_dataset(dataset_name) as (base, query, ground_truth):
#         run_benchmark(collection, base, query, ground_truth, thresholds)


@pytest.mark.parametrize(
    "collection, dataset_name, thresholds",
    [
        (
            {"dim": 100, "name": "glove", "distance": "cosine"},
            "glove",
            BenchmarkThresholds(
                min_throughput=100.0, max_avg_latency=25.0, min_recall=0.85
            ),
        )
    ],
    indirect=["collection"],
)
def test_benchmark_glove(collection, dataset_name, thresholds):
    with load_dataset(dataset_name) as (base, query, ground_truth):
        run_benchmark(collection, base, query, ground_truth, thresholds)
