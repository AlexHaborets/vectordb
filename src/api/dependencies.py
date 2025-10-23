from typing import Any, Generator
from fastapi import Request
from sqlalchemy.orm.session import Session

from src.core import VamanaIndexer
from src.db import session_manager


def get_db_session() -> Generator[Session, Any, None]:
    with session_manager.get_session() as db:
        yield db


def get_indexer(request: Request) -> VamanaIndexer:
    return request.app.state.indexer
