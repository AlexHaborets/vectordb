# Minimalistic Vector Database in Python

A lightweight, pure-Python vector database built from scratch.

This project explores the mechanics of vector similarity search by implementing a custom indexer based on the Vamana Graph algorithm ([DiskANN](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)). 
It is designed for educational purposes and lightweight use cases, including semantic search and Retrieval Augmented Generation (RAG).

## Key Features ✨

> ⚠️ **Note:** This project is a **work in progress**. Some features are incomplete and are subject to change.

-   **Vamana Graph Indexing:** Efficient approximate nearest neighbor (ANN) search using the Vamana algorithm, optimized with [Numba](https://numba.pydata.org/).  
-   **Persistence:** Robust and portable data storage using SQLite, managed via [SQLAlchemy](https://www.sqlalchemy.org/) and [Alembic](https://alembic.sqlalchemy.org/en/latest/).
-   **Collection Management:** Supports creating and managing distinct collections of vectors.
-   **RESTful API:** Fully interactive API built with [FastAPI](https://fastapi.tiangolo.com/).
-   **Python SDK:** Includes a light Python client with automatic batching and NumPy support.

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

Check out the [examples](examples/) folder in the root of the repository for detailed usage tutorials:

- [Tutorial Notebook](examples/tutorial.ipynb): Interactive guide using Pandas and HuggingFace models.

## Acknowledgements 📖

This project was built with reference to the following research and implementations:
-   **DiskANN:** Subramanya, S. J., et al. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **FreshDiskANN:** Singh, A., et al. (2021). [FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613). *arXiv preprint arXiv:2105.09613*.
-   **Vamana Visualization:** [sushrut141/vamana](https://github.com/sushrut141/vamana) - A helpful repo demonstrating the Vamana algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.