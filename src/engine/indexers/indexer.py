from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Indexer(ABC):
    @abstractmethod
    def index() -> None:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, int]]:
        pass
