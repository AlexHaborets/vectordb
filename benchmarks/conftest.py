import os
import signal
import socket
import subprocess
import time
from typing import Any, Generator

import pytest
from vectordb import Client, Collection

BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="session", autouse=True)
def server() -> Generator[None, Any, None]:
    """
    Starts the server in a separate process to ensure accurate benchmarks.
    """
    proc = subprocess.Popen(
        ["uvicorn", "src.main:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  
    )

    # Healthcheck 
    start_time = time.time()
    while time.time() - start_time < 5:
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.1):
                break
        except (OSError, ConnectionRefusedError):
            time.sleep(0.1)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise RuntimeError("Server failed to start")

    yield

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()


@pytest.fixture(scope="session")
def client() -> Generator[Client, Any, None]:
    with Client(BASE_URL) as c:
        yield c


@pytest.fixture(scope="function")
def collection(client: Client, request) -> Generator[Collection, Any, None]:
    name = "benchmark"
    dim = 384
    distance = "l2"

    if hasattr(request, "param"):
        dim = request.param.get("dim", dim)
        name = request.param.get("name", name)
        distance = request.param.get("distance", distance)

    try:
        client.delete_collection(name)
    except:  # noqa: E722
        pass

    collection = client.create_collection(name, dim, distance)

    yield collection

    client.delete_collection(name)
