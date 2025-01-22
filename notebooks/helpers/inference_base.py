from abc import ABC, abstractmethod
import numpy as np


class InferenceBase(ABC):
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
    
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass