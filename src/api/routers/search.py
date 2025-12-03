from typing import Annotated, List

from fastapi import APIRouter, Depends

from src.api.dependencies import get_indexer_manager, get_uow
from src.db import UnitOfWork
from src.engine import IndexerManager
from src.schemas import Query, SearchResult
from src.services import IndexService

search_router = APIRouter(prefix="/collections/{collection_name}", tags=["search"])

index_service = IndexService()

@search_router.post("/search", response_model=List[SearchResult])
def search(
    collection_name: str,
    q: Query,
    uow: Annotated[UnitOfWork, Depends(get_uow)],
    indexer_manager: Annotated[IndexerManager, Depends(get_indexer_manager)],
) -> List[SearchResult]:
    with uow:
        return index_service.search(
            collection_name=collection_name,
            query=q,
            indexer_manager=indexer_manager,
            uow=uow
        )

