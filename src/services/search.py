from src.db import UnitOfWork
from src.engine import VamanaIndexer
from src.schemas import Query

class SearchService:
    def __init__(self) -> None:
        pass

    def search(
        self, 
        query: Query,
        k: int,
        indexer: VamanaIndexer,
        uow: UnitOfWork
    ):
        results = indexer.search(query_vector=query.numpy_vector, k=k)

