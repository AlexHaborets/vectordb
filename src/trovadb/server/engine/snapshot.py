import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndexSnapshot:
    graph: np.ndarray
    id_map: np.ndarray
    entry_point: int


# pack N and R in 8 bytes, little endian
SNAPSHOT_HEADER = struct.Struct("<II")


def pack_snapshot(snapshot: IndexSnapshot) -> bytes:
    """
    Binary format:
    N (uint32) | R (uint32) | graph (N*R*4 bytes) | idmap (N*8 bytes)
    """
    N, R = snapshot.graph.shape
    return (
        SNAPSHOT_HEADER.pack(N, R)
        + snapshot.graph.tobytes()
        + snapshot.id_map.tobytes()
    )


def unpack_snapshot(payload: bytes, entry_point: int) -> IndexSnapshot:
    N, R = SNAPSHOT_HEADER.unpack_from(payload)
    graph_end = SNAPSHOT_HEADER.size + N * R * 4
    graph = np.frombuffer(
        payload, dtype=np.int32, count=N * R, offset=SNAPSHOT_HEADER.size
    ).reshape(N, R)
    id_map = np.frombuffer(payload, dtype=np.int64, offset=graph_end)
    return IndexSnapshot(
        graph=graph.copy(), id_map=id_map.copy(), entry_point=entry_point
    )
