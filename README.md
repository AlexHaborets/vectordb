# Minimalistic Vector Database written in Python

A lightweight, experimental vector database implementation built from scratch in Python. 

This hobby project explores the mechanics of vector similarity search by implementing a custom indexer based on the Vamana Graph algorithm ([DiskANN](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)). 
It is designed for educational purposes and lightweight use cases, including semantic search for Retrieval Augmented Generation (RAG).

## Key Features ✨

> ⚠️ **Note:** This project is a **work in progress**. Some features are incomplete and are subject to change.

-   **Vamana Graph Indexing:** Implements the Vamana graph algorithm for efficient approximate nearest neighbor (ANN) search.
-   **Collection Management:** Supports creating and managing distinct collections of vectors.
-   **RESTful API:** Fully interactive API built with [FastAPI](https://fastapi.tiangolo.com/).
-   **Persistence:** Uses a single SQLite database for metadata storage, managed via [SQLAlchemy](https://www.sqlalchemy.org/) and [Alembic](https://alembic.sqlalchemy.org/en/latest/).
-   **Python Client:** Includes a minimal client library for database interaction.

## Documentation

*Documentation regarding installation, configuration, and client usage is coming soon.*

## Acknowledgements 📖

This project was built with reference to the following research and implementations:
-   **DiskANN:** Subramanya, S. J., et al. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **Vamana Visualization:** [sushrut141/vamana](https://github.com/sushrut141/vamana) - A helpful repo demonstrating the Vamana algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.