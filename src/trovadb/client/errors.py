import functools

import httpx


class VectorDBError(Exception):
    pass


class NotFoundError(VectorDBError):
    pass


class AlreadyExistsError(VectorDBError):
    pass


class ServerError(VectorDBError):
    pass

class ConnectionError(VectorDBError):
    pass


def handle_transport_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            try:
                msg = e.response.json().get("detail", e.response.text)
            except ValueError:
                msg = e.response.text

            match e.response.status_code:
                case 404:
                    raise NotFoundError(msg) from e
                case 409:
                    raise AlreadyExistsError(msg) from e
                case _ if 400 <= e.response.status_code < 500:
                    raise VectorDBError(msg) from e
                case _:
                    raise ServerError(f"Server Error: {msg}") from e

        except httpx.RequestError as e:
            raise ConnectionError(f"Connection Error: {e}") from e

    return wrapper
