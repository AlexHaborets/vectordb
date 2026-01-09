# VectorDB Python SDK

A lightweight Python client for interacting with **VectorDB**.

## 📦 Installation

**From source (Development):**

```bash
cd sdk
pip install -e .
```

**Requirements:**
*   Python 3.9+
*   `httpx`
*   `pydantic`
*   `numpy`

## 🚀 Quick Start

Here is how to create a collection, insert vectors, and perform a semantic search.

```python
from vectordb.client import Client

# Connect to the Server
with Client("http://localhost:8000") as client:

    # Create a Collection
    collection = client.get_or_create_collection(
        name="demo_collection",
        dimension=3,
        metric="cosine"
    )

    # Add Data (Upsert)
    # The client supports native Python lists and NumPy arrays
    collection.upsert(
        ids=["1", "2"],
        vectors=[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]],
        metadatas=[{"label": "low"}, {"label": "high"}]
    )

    # Search
    results = collection.search(query=[0.1, 0.1, 0.1], k=1)

    for res in results:
        print(f"ID: {res.vector.id}, Score: {res.score}")
```

## 📂 Examples

Check the [examples](../examples/) folder in the root of the repository for detailed usage:

- [Tutorial Notebook](../examples/tutorial.ipynb): Interactive guide using Pandas and HuggingFace models.

- [Large Dataset Benchmark](../examples/rag.ipynb): A stress test loading 50,000+ DBpedia articles for RAG.
