import numpy as np
from abc import ABC, abstractmethod


class LabelStrategy(ABC):
    """
    Abstract base class that defines a strategy for processing labels for images.

    This class provides a template for implementing different methods to load and
    process labels (e.g., annotations, masks) based on file paths and image shapes.
    All subclasses must implement the `process_label` method.

    Args:
        ABC (class): Inherits from Python's Abstract Base Class (ABC) to enforce implementation of abstract methods in subclasses.
    """

    @abstractmethod
    def process_label(self, label_path: str, image_shape: tuple) -> np.array:
        """
        Abstract method to be implemented by subclasses to process image labels.

        This method should handle the loading and processing of label data from
        a given file path and adjust it to fit the provided image shape.

        Args:
            label_path (str): The file path to the label (e.g., a file containing annotations or segmentation masks).
            image_shape (tuple): The shape of the corresponding image in the form (height, width, channels).

        Returns:
            np.array: The processed label data, typically as a NumPy array that fits the given image shape.
        """
        pass
