from abc import ABC, abstractmethod
import numpy as np

class LabelStrategy(ABC):
    @abstractmethod
    def process_label(self, label_path: str, image_shape: tuple) -> np.array:
        pass