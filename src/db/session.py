from contextlib import contextmanager
from typing import Any, Generator
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session

from src.common import config


class SessionManager:
    def __init__(self, url: str) -> None:
        if not url:
            raise ValueError("SessionManager requires database url.")
        self.engine = create_engine(url)
        self.sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False)

    def get_session_factory(self) -> sessionmaker:
        return self.sessionmaker
    
    def close(self) -> None:
        self.engine.dispose()

if not config.DATABASE_URL:
    raise ValueError("DATABASE_URL env variable is missing")

session_manager = SessionManager(config.DATABASE_URL)