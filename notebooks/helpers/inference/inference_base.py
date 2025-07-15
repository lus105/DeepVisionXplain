from abc import ABC, abstractmethod
import numpy as np


class InferenceBase(ABC):
    """
    Abstract base class for inference models.

    Attributes:
        model_path (str): Path to the model file.
        model: Loaded model object (framework-dependent).
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self.model = None
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model by loading it and setting up required resources.
        """
        pass

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Perform inference on the provided data.

        Args:
            data (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Prediction results.
        """
        pass
