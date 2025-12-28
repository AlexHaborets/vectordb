from enum import Enum


class MetricType(str, Enum):
    L2 = "l2"
    COSINE = "cosine"

    def __int__(self) -> int:
        # We only convert to int when creating the indexer,
        # so this doesn't need to be BLAZINGLY fast
        metric_map = {"l2": 0, "cosine": 1}
        return metric_map[self.value]
