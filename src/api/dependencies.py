from typing import Any, Generator

from src.db import session_manager
from src.db.uow import DBUnitOfWork, UnitOfWork
from src.engine.indexer_manager import IndexerManager
from apscheduler.schedulers.asyncio import AsyncIOScheduler

_indexer_manager = IndexerManager()

def get_indexer_manager() -> IndexerManager:
    return _indexer_manager

def get_uow() -> Generator[UnitOfWork, Any, None]:
    uow = DBUnitOfWork(session_manager.get_session_factory())
    yield uow

_scheduler = AsyncIOScheduler() 

def get_scheduler() -> AsyncIOScheduler:
    return _scheduler
