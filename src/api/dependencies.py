from typing import Any, Generator
from fastapi import Request

from src.engine import VamanaIndexer
from src.db import session_manager
from src.db.uow import DBUnitOfWork, UnitOfWork

def get_indexer(request: Request) -> VamanaIndexer:
    return request.app.state.indexer

def get_uow() -> Generator[UnitOfWork, Any, None]:
    uow =  DBUnitOfWork(session_manager.get_session_factory())
    yield uow