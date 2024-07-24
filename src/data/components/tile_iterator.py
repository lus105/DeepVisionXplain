import math
import cv2
import numpy as np
from collections import defaultdict

from .tile import Tile


class TileIterator:
    def __init__(
        self,
        tile_width: int = 128,
        tile_height: int = 128,
        min_defective_area: float = 0.1,
        overlap: int = 64,
        step_size: int = 10,
        good_name: str = "0",
        defective_name: str = "1",
    ):
        """
        Initialize a TileIterator object.

        :param tile_width: Width of each tile.
        :param tile_height: Height of each tile.
        :param min_defective_area: Minimum area to be considered defective.
        :param overlap: The overlap between the tiles.
        :param good_name: Label name for non-defective tiles.
        :param defective_name: Label name for defective tiles.
        """
        self.__tile_size = (tile_width, tile_height)
        self.__min_defective_area = min_defective_area
        self.__overlap = overlap
        self.__step_size = step_size
        self._good_name = good_name
        self._defective_name = defective_name

    def _get_tiles_whole_area(
        self, image: np.array, label: np.array, image_name: str
    ) -> dict[str, list[Tile]]:
        """
        Generate and classify tiles from the given image as 'good' or 'defective'.

        :param image: Image to be tiled.
        :param label: Label for the image.
        :param image_name: Name of the image.
        :return: Dictionary of tiles categorized as 'good' or 'defective'.
        """
        height, width = image.shape[:2]
        n_patches_h = math.ceil(
            (height - self.__overlap) / (self.__tile_size[0] - self.__overlap)
        )
        n_patches_w = math.ceil(
            (width - self.__overlap) / (self.__tile_size[1] - self.__overlap)
        )

        tile_dict = defaultdict(list)
        for y_i in range(n_patches_h):
            y_st = (self.__tile_size[0] - self.__overlap) * y_i
            y_st = max(0, min(y_st, height - self.__tile_size[0]))

            for x_i in range(n_patches_w):
                x_st = (self.__tile_size[1] - self.__overlap) * x_i
                x_st = max(0, min(x_st, width - self.__tile_size[1]))

                tile = self.__create_tile(image, label, image_name, y_st, x_st)
                category = (
                    self._defective_name
                    if tile.is_defective(self.__min_defective_area)
                    else self._good_name
                )
                tile_dict[category].append(tile)

        return dict(tile_dict)

    def _get_tiles_defective_area(
        self, image: np.array, label: np.array, image_name: str
    ) -> dict[str, list]:
        """
        Generate tiles with their centers over contour points of defective areas, iterating with a defined step size.

        :param image: Image to be tiled.
        :param label: Label mask for defective areas.
        :param image_name: Name of the image.
        :return: Dictionary of tiles categorized as 'good' or 'defective'.
        """
        _, thresh = cv2.threshold(label, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tile_dict = defaultdict(list)

        for contour in contours:
            # Simplify the contour to reduce the number of points if needed
            for i in range(0, len(contour), self.__step_size):
                center_point = contour[i][0]
                center_x, center_y = center_point[0], center_point[1]

                # Calculate the top-left corner of the tile
                x_st = center_x - self.__tile_size[1] // 2
                y_st = center_y - self.__tile_size[0] // 2

                # Ensure the tile is within image boundaries
                x_st = max(0, min(x_st, image.shape[1] - self.__tile_size[1]))
                y_st = max(0, min(y_st, image.shape[0] - self.__tile_size[0]))

                tile = self.__create_tile(image, label, image_name, y_st, x_st)
                category = (
                    self._defective_name
                    if tile.is_defective(self.__min_defective_area)
                    else self._good_name
                )
                tile_dict[category].append(tile)

        return dict(tile_dict)

    def __create_tile(
        self, image: np.array, label: np.array, image_name: str, y_st: int, x_st: int
    ) -> Tile:
        """
        Creates a Tile object from a specific section of the image and label.

        :param image: Numpy array representing the full image.
        :param label: Numpy array representing the full label.
        :param image_name: Name of the image.
        :param y_st: Starting y-coordinate for the tile extraction.
        :param x_st: Starting x-coordinate for the tile extraction.
        :return: A Tile object containing the extracted image tile, label tile, image name, and rectangle coordinates.
        """
        image_tile = image[
            y_st : y_st + self.__tile_size[0], x_st : x_st + self.__tile_size[1]
        ]
        label_tile = label[
            y_st : y_st + self.__tile_size[0], x_st : x_st + self.__tile_size[1]
        ]
        rect = (x_st, y_st, x_st + self.__tile_size[1], y_st + self.__tile_size[0])

        return Tile(image_tile, label_tile, image_name, rect)
