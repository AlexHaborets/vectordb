import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api import collection_router, search_router, vector_router
from src.api.exception_handlers import create_exception_handler
from src.common.exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    InvalidOperationError,
)
from src.db import session_manager
from src.db.uow import DBUnitOfWork
from src.api.dependencies import get_indexer_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s (%(name)s): %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = logger

    app.state.logger.info("Starting vector db...")

    
    yield
    app.state.logger.info("Shutting down vector db...")

    uow = DBUnitOfWork(session_manager.sessionmaker)

    with uow:
        app.state.logger.info("Saving indexes to disk...")
        get_indexer_manager().save_all(uow)

    session_manager.close()


app = FastAPI(title="Simple VectorDB", lifespan=lifespan)

app.include_router(vector_router)
app.include_router(search_router)
app.include_router(collection_router)

app.add_exception_handler(
    EntityNotFoundError, create_exception_handler(status_code=404)
)

app.add_exception_handler(
    DuplicateEntityError, create_exception_handler(status_code=409)
)

app.add_exception_handler(
    InvalidOperationError, create_exception_handler(status_code=400)
)

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Vector DB is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8000)
