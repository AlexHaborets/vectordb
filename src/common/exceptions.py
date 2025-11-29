class VectorDBError(Exception):
    # Base class for all errors
    pass


class EntityNotFoundError(VectorDBError):
    pass


class CollectionNotFoundError(EntityNotFoundError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Collection '{name}' not found.")


class VectorNotFoundError(EntityNotFoundError):
    def __init__(self, vector_id: int) -> None:
        super().__init__(f"Vector with id {vector_id} not found")


class DuplicateEntityError(VectorDBError):
    pass


class CollectionAlreadyExistsError(DuplicateEntityError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Collection '{name}' already exists.")


class InvalidOperationError(VectorDBError):
    pass


class WrongVectorDimensionsError(InvalidOperationError):
    def __init__(self, actual_dims: int, expected_dims: int) -> None:
        super().__init__(
            f"Wrong vector dimension ({actual_dims}) for index with dimension {expected_dims}"
        )
