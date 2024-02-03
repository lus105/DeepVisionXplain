import math
import numpy as np
from collections import defaultdict

from .tile import Tile

class TileIterator:
    def __init__(self,
                 tile_width: int = 128,
                 tile_height: int = 128,
                 min_defective_area: float = 0.1,
                 overlap: int = 64,
                 good_name: str = "0",
                 defective_name: str = "1"):
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
        self.__good_name = good_name
        self.__defective_name = defective_name

    def _get_tiles(self,
                  image: np.array,
                  label: np.array,
                  image_name: str) -> dict[str, list[Tile]]:
        """
        Generate and classify tiles from the given image as 'good' or 'defective'.

        :param image: Image to be tiled.
        :param label: Label for the image.
        :param image_name: Name of the image.
        :return: Dictionary of tiles categorized as 'good' or 'defective'.
        """
        height, width = image.shape[:2]
        n_patches_h = math.ceil((height - self.__overlap) / (self.__tile_size[0] - self.__overlap))
        n_patches_w = math.ceil((width - self.__overlap) / (self.__tile_size[1] - self.__overlap))

        tile_dict = defaultdict(list)
        for y_i in range(n_patches_h):
            y_st = (self.__tile_size[0] - self.__overlap) * y_i
            y_st = max(0, min(y_st, height - self.__tile_size[0]))

            for x_i in range(n_patches_w):
                x_st = (self.__tile_size[1] - self.__overlap) * x_i
                x_st = max(0, min(x_st, width - self.__tile_size[1]))
                
                tile = self.__create_tile(image, label, image_name, y_st, x_st)
                category = self.__defective_name if tile.is_defective(self.__min_defective_area) else self.__good_name
                tile_dict[category].append(tile)

        return dict(tile_dict)

    def __create_tile(self,
                      image: np.array,
                      label: np.array,
                      image_name: str,
                      y_st: int,
                      x_st: int) -> Tile:
        """
        Creates a Tile object from a specific section of the image and label.

        :param image: Numpy array representing the full image.
        :param label: Numpy array representing the full label.
        :param image_name: Name of the image.
        :param y_st: Starting y-coordinate for the tile extraction.
        :param x_st: Starting x-coordinate for the tile extraction.
        :return: A Tile object containing the extracted image tile, label tile, image name, and rectangle coordinates.
        """
        image_tile = image[y_st:y_st + self.__tile_size[0], x_st:x_st + self.__tile_size[1]]
        label_tile = label[y_st:y_st + self.__tile_size[0], x_st:x_st + self.__tile_size[1]]
        rect = (x_st, y_st, x_st + self.__tile_size[1], y_st + self.__tile_size[0])

        return Tile(image_tile, label_tile, image_name, rect)