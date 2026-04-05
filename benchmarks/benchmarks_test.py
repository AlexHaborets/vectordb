import csv
import os
import time
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
    dataset_name: str,
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
                    "timestamp",
                    "dataset",
                    "indexing_time_s",
                    "throughput_vps",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "recall_at_10",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                dataset_name,
                f"{duration:.4f}",
                f"{throughput:.2f}",
                f"{avg_latency:.4f}",
                f"{p95_latency:.4f}",
                f"{recall:.4f}",
            ]
        )


def print_benchmark_report(
    dataset_name: str,
    duration: float,
    throughput: float,
    avg_latency: float,
    p95_latency: float,
    recall: float,
):
    """Utility to print a beautiful, readable report to the console."""
    print(f"\n{'=' * 45}")
    print(f" Benchmark Results: {dataset_name.upper()}")
    print(f"{'=' * 45}")
    print(f" Indexing Time: {duration:.2f} s")
    print(f" Throughput:    {throughput:.2f} vectors/sec")
    print(f" Avg Latency:   {avg_latency:.2f} ms")
    print(f" P95 Latency:   {p95_latency:.2f} ms")
    print(f" Recall@10:     {recall:.4f}")
    print(f"{'=' * 45}\n")


def run_benchmark(
    collection, base_vecs, query_vecs, ground_truth, thresholds: BenchmarkThresholds
):
    N_VEC = len(base_vecs)
    N_QUERIES = len(query_vecs)
    K_SEARCH = 10
    BATCH_SIZE = 5000

    # ==========================================
    # 1. WRITE PHASE (Throughput & Indexing)
    # ==========================================
    print(f"\n[Write] Upserting {N_VEC} vectors...")

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
    # 2. READ PHASE (Latency & Recall)
    # ==========================================
    print(f"\n[Read] Running {N_QUERIES} queries...")

    latencies = []
    hits = 0
    total_expected = N_QUERIES * K_SEARCH

    query_vecs_loaded = query_vecs[:]

    for i, query_vec in enumerate(query_vecs_loaded):
        # Time the search
        t0 = time.perf_counter()
        results = collection.search(query=query_vec.tolist(), k=K_SEARCH)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        # Calculate accuracy
        retrieved_ids = [int(r.vector.id) for r in results]
        true_ids = ground_truth[i][:K_SEARCH]
        hits += len(np.intersect1d(retrieved_ids, true_ids))

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    recall = hits / total_expected

    # ==========================================
    # 3. REPORTING & CSV LOGGING
    # ==========================================
    dataset_name = collection.name

    print_benchmark_report(
        dataset_name=dataset_name,
        duration=indexing_duration,
        throughput=throughput,
        avg_latency=avg_latency,
        p95_latency=p95_latency,
        recall=recall,
    )

    log_results_to_csv(
        dataset_name=dataset_name,
        duration=indexing_duration,
        throughput=throughput,
        avg_latency=avg_latency,
        p95_latency=p95_latency,
        recall=recall,
    )

    # ==========================================
    # 4. ASSERTIONS
    # ==========================================
    assert throughput >= thresholds.min_throughput, (
        f"Write too slow! Got {throughput:.2f} v/s, need {thresholds.min_throughput} v/s"
    )

    assert avg_latency <= thresholds.max_avg_latency, (
        f"Read too slow! Avg: {avg_latency:.2f}ms, Limit: {thresholds.max_avg_latency}ms"
    )

    assert recall >= thresholds.min_recall, (
        f"Recall too low! Got {recall:.4f}, need {thresholds.min_recall}"
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
