from pathlib import Path
import cv2
from dataclasses import dataclass
from tqdm.rich import tqdm
import numpy as np
from typing import Generator

from .preproc_strategy import PreprocessingStep
from ..utils import (
    list_files,
    find_file_by_name,
    IMAGE_EXTENSIONS,
    XML_EXTENSION,
    JSON_EXTENSION,
)
from ..label_parsing.label_strategy_factory import get_label_strategy
from ..label_parsing.label_strategy import LabelStrategy
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class Tile:
    image: np.array
    label: np.array
    image_name: str
    rect: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    def get_tile_name(self) -> str:
        """ "Tile name.

        Returns:
            str: Tile name based on coordinates.
        """
        return f'{self.image_name}_{self.rect[0]}_{self.rect[1]}.png'

    def get_label_tile_name(self) -> str:
        """Label tile name.

        Returns:
            str: Tile label name based on coordinates.
        """
        return f'{self.image_name}_{self.rect[0]}_{self.rect[1]}.png'

    def save_tile(self, output_dir: Path) -> None:
        """Save tile (image).

        Args:
            output_dir (Path): Path where to save tile (image).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / self.get_tile_name()), self.image)

    def save_label_tile(self, output_dir: str) -> None:
        """Save label tile (image).

        Args:
            output_dir (str): Path where to save label tile (image).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / self.get_label_tile_name()), self.label)

    def is_defective(self, defective_area_perc: float) -> bool:
        """Checks if tile contains defects.

        Args:
            defective_area_perc (float):

        Returns:
            bool: True if tile contains defects.
        """
        total_pixels = self.image.shape[0] * self.image.shape[1]
        return np.sum(self.label == 255) / total_pixels > defective_area_perc

    def is_background(self, background_perc: float) -> bool:
        """
        Determines if an image has a background percentage of black pixels
        greater than the specified threshold.

        Args:
            background_perc (float): The threshold percentage for black pixels.

        Returns:
            bool: True if the percentage of black pixels is greater than the threshold, False otherwise.
        """
        black_pixels = np.all(self.image == 0, axis=-1)
        black_pixel_count = np.sum(black_pixels)
        total_pixels = self.image.shape[0] * self.image.shape[1]
        black_pixel_percentage = black_pixel_count / total_pixels
        return black_pixel_percentage > background_perc


def sliding_window_with_coordinates(
    image: np.array, tile_size: tuple[int, int], overlap: int
) -> Generator[tuple[np.array, tuple[int, int, int, int]], None, None]:
    """
    Slide a window across the image and yield patches with their coordinates.

    Args:
        image (np.array): The input image.
        tile_size (Tuple[int, int]): The size of each tile as (height, width).
        overlap (int): The number of pixels each tile should overlap with the previous one.

    Yields:
        Generator[Tuple[np.array, Tuple[int, int, int, int]]]: A generator yielding
        each image tile along with its coordinates (x1, y1, x2, y2).
    """
    step_y = tile_size[0] - overlap
    step_x = tile_size[1] - overlap
    for y in range(0, image.shape[0], step_y):
        for x in range(0, image.shape[1], step_x):
            # Ensure the tile fits within the image bounds
            x_start = min(x, image.shape[1] - tile_size[1])
            y_start = min(y, image.shape[0] - tile_size[0])
            x_end = x_start + tile_size[1]
            y_end = y_start + tile_size[0]

            image_tile = image[y_start:y_end, x_start:x_end]
            yield image_tile, (x_start, y_start, x_end, y_end)


class TilingStep(PreprocessingStep):
    def __init__(
        self,
        tile_size: tuple[int, int] = (224, 224),
        min_defective_area_th: float = 0.1,
        discard_background_th: float = 0.0,
        overlap: int = 64,
        contour_iter_step_size: int = 10,
        iterate_over_defective_areas: bool = False,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.min_defective_area_th = min_defective_area_th
        self.overlap = overlap
        self.contour_iter_step_size = contour_iter_step_size
        self.iterate_over_defective_areas = iterate_over_defective_areas
        self.discard_background_th = discard_background_th
        self.tiles_subdir = 'tiles'
        self.good_subdir = '0'
        self.defective_subdir = '1'

        self.label_parsing_strategy: LabelStrategy = None

    def process(self, data: dict) -> dict:
        for split, path in data.items():
            log.info(f'Tiling data in {path} ...')
            self._split_images(path)
        new_data = self.get_processed_data_path(data)
        return new_data

    def get_processed_data_path(self, data: dict) -> dict:
        new_data = {}
        for split, path in data.items():
            tile_subdir = path / self.tiles_subdir / self._image_subdir
            new_data[split] = tile_subdir

        return new_data

    def _extract_tiles(
        self, image: np.array, label: np.array, image_name: str
    ) -> dict[str, list[Tile]]:
        """
        Generate and classify tiles as 'good' or 'defective' based on overlap and tile size.
        """
        tiles = {}

        for image_tile, rect in sliding_window_with_coordinates(
            image, self.tile_size, self.overlap
        ):
            label_tile = label[rect[1] : rect[3], rect[0] : rect[2]]
            tile = Tile(image_tile, label_tile, image_name, rect)
            category = (
                self.defective_subdir
                if tile.is_defective(self.min_defective_area_th)
                else self.good_subdir
            )
            tiles.setdefault(category, []).append(tile)

        return tiles

    def _extract_defective_tiles(
        self, image: np.array, label: np.array, image_name: str
    ) -> dict[str, list[Tile]]:
        """
        Generate tiles centered over defective area contours.
        """
        tiles = {}
        contours, _ = cv2.findContours(
            label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            for i in range(0, len(contour), self.contour_iter_step_size):
                center_x, center_y = contour[i][0][0], contour[i][0][1]

                x_st = max(center_x - self.tile_size[1] // 2, 0)
                y_st = max(center_y - self.tile_size[0] // 2, 0)

                x_st = min(x_st, image.shape[1] - self.tile_size[1])
                y_st = min(y_st, image.shape[0] - self.tile_size[0])

                tile = Tile(
                    image[
                        y_st : y_st + self.tile_size[0], x_st : x_st + self.tile_size[1]
                    ],
                    label[
                        y_st : y_st + self.tile_size[0], x_st : x_st + self.tile_size[1]
                    ],
                    image_name,
                    (x_st, y_st, x_st + self.tile_size[1], y_st + self.tile_size[0]),
                )
                category = (
                    self.defective_subdir
                    if tile.is_defective(self.min_defective_area_th)
                    else self.good_subdir
                )
                tiles.setdefault(category, []).append(tile)

        return tiles

    def _split_images(self, path: Path):
        image_paths = list_files(
            path / self._image_subdir, file_extensions=IMAGE_EXTENSIONS
        )
        tile_image_subdir, tile_label_subdir = self._prepare_tile_directories(path)

        for image_path in tqdm(image_paths, desc='Processing images'):
            image = cv2.imread(str(image_path))

            label_path = find_file_by_name(
                path / self._label_subdir,
                image_path.stem,
                file_extensions=IMAGE_EXTENSIONS + [XML_EXTENSION, JSON_EXTENSION],
            )

            if self.label_parsing_strategy is None:
                self.label_parsing_strategy = get_label_strategy(label_path.suffix)

            label = self.label_parsing_strategy.process_label(label_path, image.shape)

            self._process_and_save_tiles(
                image, label, image_path.stem, tile_image_subdir, tile_label_subdir
            )

            if self.iterate_over_defective_areas:
                self._process_and_save_defective_tiles(
                    image, label, image_path.stem, tile_image_subdir, tile_label_subdir
                )

    def _prepare_tile_directories(self, path: Path) -> tuple:
        tile_subdir = path / self.tiles_subdir
        tile_image_subdir = tile_subdir / self._image_subdir
        tile_label_subdir = tile_subdir / self._label_subdir

        tile_image_subdir.mkdir(parents=True, exist_ok=True)
        tile_label_subdir.mkdir(parents=True, exist_ok=True)

        return tile_image_subdir, tile_label_subdir

    def _process_and_save_tiles(
        self, image, label, image_name, tile_image_subdir, tile_label_subdir
    ):
        tiles = self._extract_tiles(image, label, image_name)
        self._save_tiles(tiles, tile_image_subdir, tile_label_subdir)

    def _process_and_save_defective_tiles(
        self, image, label, image_name, tile_image_subdir, tile_label_subdir
    ):
        defective_tiles = self._extract_defective_tiles(image, label, image_name)
        self._save_tiles(defective_tiles, tile_image_subdir, tile_label_subdir)

    def _save_tiles(self, tiles, tile_image_subdir, tile_label_subdir):
        for category, tile_list in tiles.items():
            for tile in tile_list:
                if not tile.is_background(self.discard_background_th):
                    tile.save_tile(tile_image_subdir / category)
                    tile.save_label_tile(tile_label_subdir / category)
