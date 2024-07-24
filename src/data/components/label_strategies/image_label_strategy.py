import cv2
import numpy as np

from .label_strategy import LabelStrategy


class ImageLabelStrategy(LabelStrategy):
    def process_label(self, label_path: str, image_shape: tuple) -> np.array:
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        _, thresholded_label = cv2.threshold(label, 5, 255, cv2.THRESH_BINARY)

        return thresholded_label
