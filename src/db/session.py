from typing import Any, Generator
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session

from src import config

class SessionManager:
    def __init__(self, url: str):
        if not url:
            raise ValueError("SessionManager requires database url.")
        self.engine = create_engine(url)
        self.sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False)

    def close(self):
        self.engine.dispose()

    def get_session(self) -> Generator[Session, Any, None]:
        session = self.sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

if not config.DATABASE_URL:
    raise ValueError("DATABASE_URL env variable is missing")

session_manager = SessionManager(config.DATABASE_URL)