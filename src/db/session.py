from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, event

from src.common import config


def configure_db(connection, connection_record) -> None:
    """
    Some optimizations for the sqlite database
    """
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class SessionManager:
    def __init__(self, url: str) -> None:
        if not url:
            raise ValueError("SessionManager requires database url.")
        
        # check_same_thread is set to False for multithreading
        self.engine = create_engine(url, connect_args={"check_same_thread": False})

        event.listen(target=self.engine, identifier="connect", fn=configure_db)

        self.sessionmaker = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

    def get_session_factory(self) -> sessionmaker:
        return self.sessionmaker

    def close(self) -> None:
        self.engine.dispose()


if not config.DATABASE_URL:
    raise ValueError("DATABASE_URL env variable is missing")

session_manager = SessionManager(config.DATABASE_URL)
