import time
import uuid
import numpy as np

MIN_WRITE_SPEED = 700.0   # vectors per second
MAX_READ_LATENCY = 15.0   # milliseconds

N_VECTORS = 10000
DIMS = 384
BATCH_SIZE = 512

def test_write_speed(collection):
    """
    Benchmarks vector upsert speed
    """

    print(f"\n[Write] Generating {N_VECTORS} vectors...")
    vectors = np.random.rand(N_VECTORS, DIMS).astype("float32")
    ids = [str(uuid.uuid4()) for _ in range(N_VECTORS)]

    start = time.perf_counter()
    collection.upsert(
        ids=ids, 
        vectors=vectors, 
        batch_size=BATCH_SIZE
    )
    duration = time.perf_counter() - start

    throughput = N_VECTORS / duration

    print(f"[Write] Time: {duration:.4f}s | Speed: {throughput:.2f} vec/s")
    assert throughput > MIN_WRITE_SPEED, \
        f"Write too slow! Got {throughput:.2f} vec/s, need > {MIN_WRITE_SPEED}"


def test_read_latency(collection):
    """
    Benchmarks search latency
    """

    print(f"\n[Read] Upserting {N_VECTORS} vectors to DB...")
    
    vectors = np.random.rand(N_VECTORS, DIMS).astype("float32")
    ids = [str(uuid.uuid4()) for _ in range(N_VECTORS)]
    
    collection.upsert(
        ids=ids, 
        vectors=vectors, 
        batch_size=BATCH_SIZE
    )

    N_QUERIES = 100
    K_SEARCH = 10
    queries = np.random.rand(N_QUERIES, DIMS).astype("float32")
    latencies = []

    print(f"[Read] Running {N_QUERIES} queries...")
    
    for i in range(N_QUERIES):
        t0 = time.perf_counter()
        
        collection.search(query=queries[i].tolist(), k=K_SEARCH)
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"[Read] Avg Latency: {avg_latency:.2f} ms")
    print(f"[Read] P95 Latency: {p95_latency:.2f} ms")

    assert avg_latency < MAX_READ_LATENCY, \
        f"Read too slow! Avg: {avg_latency:.2f}ms, Limit: {MAX_READ_LATENCY}ms"