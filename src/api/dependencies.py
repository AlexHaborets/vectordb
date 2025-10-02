from src.db import session_manager

def get_db():
    with session_manager.get_session() as db:
        yield db