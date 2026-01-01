import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from src.api import collection_router, search_router, vector_router
from src.api.dependencies import get_indexer_manager, get_scheduler
from src.api.exception_handlers import create_exception_handler
from src.common import (
    DuplicateEntityError,
    EntityNotFoundError,
    InvalidOperationError,
    setup_logger,
)
from src.common.config import AUTO_SAVE_INDEX_PERIOD
from src.db import session_manager
from apscheduler.triggers.interval import IntervalTrigger

from src.db.uow import DBUnitOfWork


def _save_indexers() -> None:
    indexer_manager = get_indexer_manager()
    uow = DBUnitOfWork(session_manager.get_session_factory())
    with uow:
        indexer_manager.save_all(uow)


async def save_indexers() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _save_indexers)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting vector db...")
    setup_logger()

    scheduler = get_scheduler()
    scheduler.add_job(
        func=save_indexers,
        id="auto_save_indexers",
        trigger=IntervalTrigger(seconds=AUTO_SAVE_INDEX_PERIOD),
        replace_existing=True,
    )

    scheduler.start()

    logger.info("Vector db started successfully")

    yield
    logger.info("Shutting down vector db...")

    scheduler.shutdown(wait=False)

    logger.info("Saving indexes to disk...")
    await save_indexers()

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
