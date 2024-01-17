import os
import cv2
import numpy as np
from typing import Tuple


class Tile:
    def __init__(self,
                 image: np.array,
                 label: np.array,
                 image_name: str,
                 rect: tuple):
        """
        Initialize a Tile object.

        :param image: numpy array representing the image.
        :param label: numpy array representing the label.
        :param image_name: name of the image.
        :param rect: a tuple (x1, y1, x2, y2) defining the rectangle.
        """
        self.__tile_image = image
        self.__tile_label = label
        self.__image_name = image_name
        self.__rect = rect

    @property
    def image(self) -> np.array:
        """Return the tile image."""
        return self.__tile_image

    @property
    def label(self) -> np.array:
        """Return the tile label."""
        return self.__tile_label

    @property
    def image_name(self) -> str:
        """Return the image name."""
        return self.__image_name

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Return the rectangle."""
        return self.__rect

    def get_tile_name(self) -> str:
        """Generate and return the tile name."""
        return f'{self.__image_name}_{self.__rect[0]}_{self.__rect[1]}.png'

    def get_label_tile_name(self) -> str:
        """Generate and return the label tile name."""
        return f'{self.__image_name}_{self.__rect[0]}_{self.__rect[1]}_mask.png'

    def get_tile_path(self, output_dir: str) -> str:
        """Return the full path for the tile image."""
        return os.path.join(output_dir, self.get_tile_name())

    def get_label_tile_path(self, output_dir: str) -> str:
        """Return the full path for the label tile."""
        return os.path.join(output_dir, self.get_label_tile_name())

    def save_tile(self, output_dir: str) -> None:
        """Save the tile image to the specified directory."""
        cv2.imwrite(self.get_tile_path(output_dir), self.__tile_image)

    def save_label_tile(self, output_dir: str) -> None:
        """Save the label tile to the specified directory."""
        cv2.imwrite(self.get_label_tile_path(output_dir), self.__tile_label)