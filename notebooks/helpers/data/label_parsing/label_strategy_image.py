import cv2
import numpy as np

from .label_strategy import LabelStrategy


class ImageLabelStrategy(LabelStrategy):
    def process_label(self, label_path: str, image_shape: tuple) -> np.array:
        """Reads image label.

        Args:
            label_path (str): The file path to the label.
            image_shape (tuple): Corresponding image shape.

        Returns:
            np.array: The processed label data as image.
        """
        if label_path is None:
            thresholded_label = np.zeros(image_shape[:2], dtype=np.uint8)
        else:
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            _, thresholded_label = cv2.threshold(
                label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return thresholded_label
