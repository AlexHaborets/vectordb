from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class MetricType(str, Enum):
    L2 = "l2"
    COSINE = "cosine"

class DistanceMetric(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        pass

class CosineDistance(DistanceMetric):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            return 1.0
        similarity = dot_product / (norm_x * norm_y)

        return 1.0 - similarity

class L2Distance(DistanceMetric):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0.0

def get_metric(type: str) -> DistanceMetric:
    match type:
        case MetricType.COSINE:
            return CosineDistance()
        case MetricType.L2:
            return L2Distance()
        case _:
            raise ValueError(f"Unknown metric: {type}")