from typing import Any, Generator

from sqlalchemy.orm import Session

from src.db import session_manager


def get_db_session() -> Generator[Session, Any, None]:
    yield from session_manager.get_session()
