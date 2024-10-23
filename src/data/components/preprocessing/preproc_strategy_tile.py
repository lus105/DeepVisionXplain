import os
import cv2
import math
from tqdm import tqdm
import numpy as np
from typing import Tuple
from collections import defaultdict
from .preproc_strategy import PreprocessingStep
from ..utils import get_file_paths_rec, get_file_name, find_annotation_file
from ..label_parsing.label_strategy_factory import get_label_strategy

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class Tile:
    def __init__(self, image: np.array, label: np.array, image_name: str, rect: tuple):
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
        return f"{self.__image_name}_{self.__rect[0]}_{self.__rect[1]}.png"

    def get_label_tile_name(self) -> str:
        """Generate and return the label tile name."""
        return f"{self.__image_name}_{self.__rect[0]}_{self.__rect[1]}_mask.png"

    def get_tile_path(self, output_dir: str) -> str:
        """Return the full path for the tile image."""
        return os.path.join(output_dir, self.get_tile_name())

    def get_label_tile_path(self, output_dir: str) -> str:
        """Return the full path for the label tile."""
        return os.path.join(output_dir, self.get_label_tile_name())

    def save_tile(self, output_dir: str) -> None:
        """Save the tile image to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(self.get_tile_path(output_dir), self.__tile_image)

    def save_label_tile(self, output_dir: str) -> None:
        """Save the label tile to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(self.get_label_tile_path(output_dir), self.__tile_label)

    def is_defective(self, min_defective_area) -> bool:
        """Return True if the tile is defective, False otherwise."""
        defective_area = np.sum(self.__tile_label == 255)
        return defective_area > min_defective_area


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


class TilingProcessor(TileIterator):
    def __init__(
        self,
        tile_width: int = 128,
        tile_height: int = 128,
        min_defective_area: float = 0.1,
        overlap: int = 64,
        step_size: int = 10,
        iterate_over_defective_areas: bool = False,
        save_every_second_good_tile: bool = False,
        images_dir: str = "images",
        labels_dir: str = "labels",
        tiles_dir: str = "tiles",
    ):
        """
        Initialize TilingProcessor with tile dimensions, minimum defective area, overlap,
        and directory names for images, labels, and tiles.
        """
        super().__init__(
            tile_width=tile_width,
            tile_height=tile_height,
            min_defective_area=min_defective_area,
            overlap=overlap,
            step_size=step_size,
        )
        self.iterate_over_defective_areas = iterate_over_defective_areas
        self.save_every_second_good_tile = save_every_second_good_tile
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.tiles_dir = tiles_dir

    def process(self, path: str, use_tqdm: bool = True) -> str:
        """
        Process all images in the given directory, creating tiles and corresponding labels.
        """
        image_list = get_file_paths_rec(os.path.join(path, self.images_dir))
        label_source_dir = os.path.join(path, self.labels_dir)
        tile_images_dir = os.path.join(path, self.tiles_dir, self.images_dir)
        tile_labels_dir = os.path.join(path, self.tiles_dir, self.labels_dir)

        for directory in [tile_images_dir, tile_labels_dir]:
            if os.path.exists(directory) and os.listdir(directory):
                log.warning(f"Directory {directory} is not empty. Skipping tiling.")
                return tile_images_dir

        progress = (
            tqdm(image_list, desc="Processing images: " + path)
            if use_tqdm
            else image_list
        )
        for image_path in progress:
            image_name = get_file_name(image_path)
            image = cv2.imread(str(image_path))
            label = self.__get_label(image, image_name, label_source_dir)
            tiles_whole_area = self._get_tiles_whole_area(image, label, image_name)
            self.__save_tiles(tiles_whole_area, tile_images_dir, tile_labels_dir)

            if self.iterate_over_defective_areas:
                tiles_defective_area = self._get_tiles_defective_area(
                    image, label, image_name
                )
                self.__save_tiles(
                    tiles_defective_area, tile_images_dir, tile_labels_dir
                )

        return tile_images_dir

    def __get_label(
        self, image: np.array, image_name: str, label_source_dir: str
    ) -> np.array:
        """
        Retrieve the label for the given image, or create a blank label if none exists.
        """
        label_path = find_annotation_file(label_source_dir, image_name)
        if label_path is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        file_extension = os.path.splitext(label_path)[-1].lower()
        label_strategy = get_label_strategy(file_extension)

        return label_strategy.process_label(label_path, image.shape)

    def __save_tiles(
        self, tiles: dict, tile_images_dir: str, tile_labels_dir: str
    ) -> None:
        """
        Save the generated tiles and their labels to the respective directories.
        """
        for key, tile_list in tiles.items():
            for i, tile in enumerate(tile_list):
                if (
                    key == self._good_name
                    and "train" in tile_images_dir
                    and self.save_every_second_good_tile
                    and i % 2 != 0
                ):
                    continue

                tile.save_tile(os.path.join(tile_images_dir, key))
                tile.save_label_tile(os.path.join(tile_labels_dir, key))


class TilingStep(PreprocessingStep):
    def __init__(
        self,
        tile_width=128,
        tile_height=128,
        min_defective_area=0.1,
        overlap=64,
        step_size=10,
        iterate_over_defective_areas=False,
        save_every_second_good_tile=False,
        images_dir="images",
        labels_dir="labels",
        tiles_dir="tiles",
    ):
        self.tiling_processor = TilingProcessor(
            tile_width=tile_width,
            tile_height=tile_height,
            min_defective_area=min_defective_area,
            overlap=overlap,
            step_size=step_size,
            iterate_over_defective_areas=iterate_over_defective_areas,
            save_every_second_good_tile=save_every_second_good_tile,
            images_dir=images_dir,
            labels_dir=labels_dir,
            tiles_dir=tiles_dir,
        )

    def process(self, data: dict) -> dict:
        # Assuming 'split_dirs' contains the paths to train, test, val directories
        for split, path in data['split_dirs'].items():
            log.info(f"Tiling data in {split} set...")
            tiled_dir = self.tiling_processor.process(str(path))
            data['tiled_dirs'][split] = tiled_dir
        return data