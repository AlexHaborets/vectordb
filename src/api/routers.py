from fastapi import APIRouter

from src.schemas import SearchResult, Vector, VectorCreate

vector_router = APIRouter(prefix="/vectors", tags=["vectors"])

search_router = APIRouter(prefix="/vectors", tags=["vectors"])


@vector_router.post("/add", response_model=Vector)
async def add(vector: VectorCreate):
    pass


@search_router.post("/", response_model=SearchResult)
async def search():
    pass
