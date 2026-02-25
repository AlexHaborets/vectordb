# Minimalistic Vector Database in Python

A lightweight, pure-Python vector database built from scratch.

This project explores the mechanics of vector similarity search by implementing a custom indexer based on the Vamana Graph algorithm ([DiskANN](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)). 
It is designed for educational purposes and lightweight use cases, including semantic search and Retrieval Augmented Generation (RAG).

## Key Features ✨

> ⚠️ **Note:** This project is a **work in progress**. Some features are incomplete and are subject to change.

- **🧭 Vamana Graph Indexing:** Implements the algorithm behind [DiskANN](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). Vamana optimizes the graph with **long-range shortcuts**, allowing the search to navigate huge datasets efficiently by jumping quickly toward the target rather than stepping slowly between neighbors.
- **⚡ Pure Python, C-Level Speed:** By leveraging **[Numba](https://numba.pydata.org/)** JIT compilation, this project achieves indexing and search performance comparable to C while maintaining a readable, hackable Python codebase.
- **💾 Persistence:** Data isn't just dumped into binary files. Metadata and vectors are stored reliably in **SQLite**, orchestrated by **[SQLAlchemy](https://www.sqlalchemy.org/)** and **[Alembic](https://alembic.sqlalchemy.org/en/latest/)**, ensuring portability and crash-safety.
- **🐍 Data Science Ready SDK:** A lightweight client designed for the Python ecosystem. It supports NumPy arrays natively and handles automatic request batching behind the scenes to maximize throughput.
- **🔌 Modern REST API:** Powered by **[FastAPI](https://fastapi.tiangolo.com/)**, providing asynchronous request handling, rigorous type safety, and automatic interactive documentation (Swagger UI).

## 📊 Benchmarks

| Metric | Dataset | Result |
| :--- | :--- | :--- |
| **Indexing Throughput** | 10k vectors (384d) | **1210.88 vec/s** |
| **Query Latency (Avg)** | 10k vectors (384d) | **6.30 ms** |
| **Query Latency (P95)** | 10k vectors (384d) | **9.35 ms** |
| **Recall@10** | SIFT-Small (10k vectors, 128d) | **0.9960** |
| **Recall@10** | SIFT1M (1M vectors, 128d) | **0.9346** |


These benchmarks were achieved using the following Vamana graph indexing config:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `VAMANA_R` | `32` | Maximum graph degree |
| `VAMANA_L_BUILD` | `80` | Search list size during index building |
| `VAMANA_L_SEARCH` | `60` | Search list size during querying |
| `VAMANA_ALPHA_FIRST_PASS` | `1.0` | Distance multiplier (First pass) |
| `VAMANA_ALPHA_SECOND_PASS` | `1.2` | Distance multiplier (Second pass) |

## 🚀 Getting Started

#### Option A: Using Docker (Recommended)
This is the fastest way to get the server running without installing dependencies.

```bash
docker compose up --build
```

#### Option B: Local Development

If you prefer running the server natively:

1. Install Dependencies:
```bash    
pip install -r requirements.txt
```

2. Start the Server:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at the default url http://localhost:8000.

### 2. Install the Client

Install the Python SDK from the sdk directory:

```bash
pip install -e sdk/
```

### 3. Client Usage

For detailed documentation, see the [SDK README](sdk/README.md).

```python
from vectordb.client import Client

with Client() as client:
    # Create
    collection = client.get_or_create_collection("demo", dimension=3, metric="cosine")
    
    # Insert
    collection.upsert(
        ids=["1", "2", "3"], 
        vectors=[
            [0.1, 0.2, 0.3], 
            [0.9, 0.8, 0.7],
            [0.2, 0.4, 0.4]
        ]
    )
    
    # Search
    results = collection.search(query=[0.1, 0.2, 0.3])
    print(results)
```

## 📂 Examples

Check out the [examples](examples/) folder in the root of the repository for detailed usage:

- [Tutorial Notebook](examples/tutorial.ipynb): Interactive guide using Pandas and HuggingFace models.

- [Large Dataset Benchmark](examples/rag.ipynb): A stress test loading 50,000+ DBpedia articles for RAG.

## Acknowledgements 📖

This project was built with reference to the following research and implementations:
-   **DiskANN:** Subramanya, S. J., et al. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **FreshDiskANN:** Singh, A., et al. (2021). [FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613). *arXiv preprint arXiv:2105.09613*.
-   **Vamana Visualization:** [sushrut141/vamana](https://github.com/sushrut141/vamana) - A helpful repo demonstrating the Vamana algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
