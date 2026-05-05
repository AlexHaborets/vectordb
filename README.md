# TrovaDB

A lightweight, pure-Python vector database built from scratch.

This project explores the mechanics of vector similarity search by implementing a custom indexer based on the Vamana Graph algorithm ([DiskANN](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)). 
Designed for educational purposes and lightweight use cases, including semantic search and Retrieval Augmented Generation (RAG).

## Key Features

> Note: This project is work in progress. APIs and features are subject to change.

- **Vamana Graph Indexing:** Utilizes the algorithm behind **[DiskANN](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)**. 
- **Index Auto-Tuning:** Implements adaptive tuning of the parameter alpha to stabilize average graph degree via a custom PI controller, fitting to different dataset structure and improving recall without sacrificing latency.
- **Built-in Reranking:** Natively supports [MMR (Maximal Marginal Relevance)](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) reranking out of the box, guaranteeing varied and contextually rich context for RAG applications.
- **C-Level Speed:** By leveraging **[Numba](https://numba.pydata.org/)** JIT compilation, TrovaDB achieves indexing and search performance comparable to C while maintaining a readable, hackable Python codebase.
- **Persistence:** The full database is stored reliably in a single SQLite file ensuring portability and crash-safety.
- **Data Science Ready SDK:** A lightweight Python client designed with native NumPy support and simple interface.
- **Familiar Stack:** Powered by **[FastAPI](https://fastapi.tiangolo.com/)**, **[SQLAlchemy](https://www.sqlalchemy.org/)** and **[Alembic](https://alembic.sqlalchemy.org/en/latest/)**.

## Getting Started

### 1. Installation

You can install TrovaDB directly from GitHub using `pip`. 

#### Option A: Client and Server (Recommended)
If you want to run the database server locally, install it with the `[server]` extra:
```bash
pip install "trovadb[server] @ git+https://github.com/AlexHaborets/trovadb.git"
```

#### Option B: Client Only
```bash
pip install git+https://github.com/AlexHaborets/trovadb.git
```

### 2. Starting the Server

Once installed with the `[server]` extra, you can easily start the database server:

```bash
trovadb-server
```
*(Runs on `localhost:8000` by default)*

#### Alternative: Using Docker  
If you prefer not to install dependencies locally, you can clone the repository and run it instantly via Docker:

```bash
docker compose up --build
```

### 3. Client Usage

Here is a quick example of how to connect to the server, upsert vectors, and perform a search:

```python
from trovadb.client import Client

with Client() as client:
    # Create a collection
    collection = client.get_or_create_collection("demo", dimension=3, metric="cosine")
    
    # Upsert vectors (combines insert & update operations in one)
    collection.upsert(
        ids=["1", "2", "3", "4", "5"], 
        vectors=[
            [0.1, 0.2, 0.3], 
            [0.9, 0.8, 0.7],
            [0.2, 0.4, 0.4],
            [0.1, 0.8, 0.2],
            [0.5, 0.3, 0.6]
        ]
    )
    
    q = [0.1, 0.2, 0.3]
    
    # Search for three nearest neighbors of q
    results = collection.search(query=q, k=3)
    print(results)

    # Delete specified vectors
    collection.delete(ids=["1", "3"])

    # Delete entire collection
    client.delete_collection("demo")
```

## Examples

Check out the [examples](examples/) folder in the root of the repository for detailed usage:

- [Tutorial Notebook](examples/tutorial.ipynb): Interactive guide using Pandas and HuggingFace models.

- [Large Dataset Benchmark](examples/rag.ipynb): A stress test loading 50,000+ DBpedia articles for RAG.

## Roadmap

**Completed Milestones**
- [X] Core vector store functionality 
- [X] Persistence
- [X] Docker support
- [X] Benchmarking test suite

**Must-Haves**
- [ ] Comprehensive test suite and unit tests
- [ ] In-memory client for local prototyping

**Future Features**
- [ ] Metadata filtering
- [ ] Hybrid search support

## Why "TrovaDB"?

The name is inspired by the italian phrase *"Cerca Trova"* ("Seek and you shall find") — a cryptic clue left by the artist Giorgio Vasari, believed to indicate that a lost [Da Vinci mural](https://en.wikipedia.org/wiki/The_Battle_of_Anghiari_(Leonardo)) is hidden beneath his fresco in Florence. It felt like a fitting name for a fast and simple search tool.

## Acknowledgements

-   **DiskANN:** Subramanya, S. J., et al. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **FreshDiskANN:** Singh, A., et al. (2021). [FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613). *arXiv preprint arXiv:2105.09613*.
-   **MMR:** Carbonell, J., & Goldstein, J. (1998). [The Use of MMR, Diversity-Based Reranking for Reordering
Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf).  *SIGIR '98*.
-   **Vamana Visualization:** [sushrut141/vamana](https://github.com/sushrut141/vamana) - A helpful repo demonstrating the core algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
