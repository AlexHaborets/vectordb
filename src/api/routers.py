from typing import Annotated, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.api.exceptions import VectorNotFoundError
from src.db.crud import CRUDRepository
from src.schemas import SearchResult, Vector, VectorCreate

vector_router = APIRouter(prefix="/vectors", tags=["vectors"])

search_router = APIRouter(prefix="/vectors", tags=["vectors"])


@vector_router.post("/", response_model=Vector, status_code=201)
def add(vector: VectorCreate, db: Annotated[Session, Depends(get_db_session)]):
    return CRUDRepository(db).add_vector(vector)


@vector_router.get("/", response_model=Vector)
def get_by_id(vector_id: int, db: Annotated[Session, Depends(get_db_session)]):
    vector = CRUDRepository(db).get_vector_by_id(vector_id)
    if not vector:
        raise VectorNotFoundError(vector_id)
    return vector


@vector_router.get("/all", response_model=List[Vector])
def get_all(db: Annotated[Session, Depends(get_db_session)]):
    return CRUDRepository(db).get_all_vectors()


@vector_router.delete("/", response_model=Vector, status_code=201)
def delete_by_id(vector_id: int, db: Annotated[Session, Depends(get_db_session)]):
    vector = CRUDRepository(db).mark_vector_deleted(vector_id)
    if not vector:
        raise VectorNotFoundError(vector_id)
    return vector

@search_router.post("/", response_model=SearchResult)
def search():
    pass
