from src.common.config import VECTOR_DIMENSIONS

class VectorNotFoundError(Exception):
    def __init__(self, vector_id: int) -> None:
        super().__init__(f"Vector with id [{vector_id}] not found")

class WrongVectorDimensionsError(Exception):
    def __init__(self, vector_dims: int) -> None:
        super().__init__(f"Wrong vector dimension ({vector_dims}) for index with dimension [{VECTOR_DIMENSIONS}]")

class CollectionNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Collection '{name}' not found.")

class CollectionAlreadyExistsError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Collection '{name}' already exists.")