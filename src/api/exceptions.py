from typing import Dict
from fastapi import HTTPException, status

from src.config import VECTOR_DIMENSIONS


class BaseAPIError(HTTPException):
    def __init__(
        self, status_code: int, detail: str, headers: Dict[str, str] | None = None
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class VectorNotFoundError(BaseAPIError):
    def __init__(self, vector_id: int):
        super().__init__(status.HTTP_404_NOT_FOUND, f"Vector with id [{vector_id}] not found")

class WrongVectorDimensionsError(BaseAPIError):
    def __init__(self, vector_dims: int):
        super().__init__(status.HTTP_404_NOT_FOUND, f"Wrong vector dimension ({vector_dims}) for index with dimension [{VECTOR_DIMENSIONS}]")
