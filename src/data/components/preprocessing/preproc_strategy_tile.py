from pathlib import Path
import cv2
from dataclasses import dataclass
from tqdm.rich import tqdm
import numpy as np
from typing import Generator

from .preproc_strategy import PreprocessingStep
from ..utils import (
    list_files,
    find_annotation_file,
    IMAGE_EXTENSIONS,
    XML_EXTENSION,
    JSON_EXTENSION
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
    rect: tuple[int, int, int, int] # (x1, y1, x2, y2)

    def get_tile_name(self) -> str:
        """"Tile name.

        Returns:
            str: Tile name based on coordinates.
        """
        return f'{self.image_name}_{self.rect[0]}_{self.rect[1]}.png'

    def get_label_tile_name(self) -> str:
        """Label tile name.

        Returns:
            str: Tile label name based on coordinates.
        """
        return f'{self.image_name}_{self.rect[0]}_{self.rect[1]}_label.png'

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

    def is_defective(self, min_defective_area: float) -> bool:
        """Checks if tile contains defects.

        Args:
            min_defective_area (float):  

        Returns:
            bool: True if tile contains defects.
        """
        return np.sum(self.label == 255) > min_defective_area


def sliding_window_with_coordinates(
        image: np.array,
        tile_size: tuple[int, int],
        overlap: int
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
    for y in range(0, image.shape[0] - tile_size[0] + 1, step_y):
        for x in range(0, image.shape[1] - tile_size[1] + 1, step_x):
            image_tile = image[y:y + tile_size[0], x:x + tile_size[1]]
            yield image_tile, (x, y, x + tile_size[1], y + tile_size[0])


class TilingStep(PreprocessingStep):
    def __init__(
        self,
        tile_size: tuple[int, int] = (224, 224),
        min_defective_area: float = 0.1,
        overlap: int = 64,
        contour_iter_step_size: int = 10,
        iterate_over_defective_areas: bool = False,
        tiles_subdir: str = 'tiles',
    ):
        super().__init__()
        self.tile_size = tile_size
        self.min_defective_area = min_defective_area
        self.overlap = overlap
        self.contour_iter_step_size = contour_iter_step_size
        self.iterate_over_defective_areas = iterate_over_defective_areas
        self.tiles_subdir = tiles_subdir

        self.label_parsing_strategy: LabelStrategy = None

    def _extract_tiles(self, image: np.array, label: np.array, image_name: str) -> dict[str, list[Tile]]:
        """
        Generate and classify tiles as 'good' or 'defective' based on overlap and tile size.
        """
        tiles = {}

        for image_tile, rect in sliding_window_with_coordinates(image, self.tile_size, self.overlap):
            label_tile = label[rect[1]:rect[3], rect[0]:rect[2]]
            tile = Tile(image_tile, label_tile, image_name, rect)
            category = 'defective' if tile.is_defective(self.min_defective_area) else 'good'
            tiles.setdefault(category, []).append(tile)
            
        return tiles

    def _extract_defective_tiles(self, image: np.array, label: np.array, image_name: str) -> dict[str, list[Tile]]:
        """
        Generate tiles centered over defective area contours.
        """
        tiles = {}
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            for i in range(0, len(contour), self.contour_iter_step_size):
                center_x, center_y = contour[i][0][0], contour[i][0][1]
                x_st = max(center_x - self.tile_size[1] // 2, 0)
                y_st = max(center_y - self.tile_size[0] // 2, 0)
                tile = Tile(
                    image[y_st: y_st + self.tile_size[0], x_st: x_st + self.tile_size[1]],
                    label[y_st: y_st + self.tile_size[0], x_st: x_st + self.tile_size[1]],
                    image_name,
                    (x_st, y_st, x_st + self.tile_size[1], y_st + self.tile_size[0])
                )
                category = 'defective' if tile.is_defective(self.min_defective_area) else 'good'
                tiles.setdefault(category, []).append(tile)

        return tiles

    def _split_images(self, path: Path, overwrite_data: bool) -> Path:
        image_paths = list_files(path / self._image_subdir, file_extensions=IMAGE_EXTENSIONS)
        tile_image_subdir, tile_label_subdir = self._prepare_tile_directories(path)

        if not overwrite_data:
            return tile_image_subdir.parent

        for image_path in tqdm(image_paths, desc='Processing images'):
            image = cv2.imread(str(image_path))

            label_path = find_annotation_file(
                path / self._label_subdir, image_path.stem,
                file_extensions=[IMAGE_EXTENSIONS, XML_EXTENSION, JSON_EXTENSION]
            )
            
            if self.label_parsing_strategy is None:
                self.label_parsing_strategy = get_label_strategy(label_path.suffix)
            
            label = self.label_parsing_strategy.process_label(label_path, image.shape)

            self._process_and_save_tiles(image, label, image_path.name, tile_image_subdir, tile_label_subdir)

            if self.iterate_over_defective_areas:
                self._process_and_save_defective_tiles(image, label, image_path.name, tile_image_subdir, tile_label_subdir)

        return tile_image_subdir.parent

    def _prepare_tile_directories(self, path: Path) -> tuple:
        tile_subdir = path / self.tiles_subdir
        tile_image_subdir = tile_subdir / self._image_subdir
        tile_label_subdir = tile_subdir / self._label_subdir

        tile_image_subdir.mkdir(parents=True, exist_ok=True)
        tile_label_subdir.mkdir(parents=True, exist_ok=True)
        
        return tile_image_subdir, tile_label_subdir

    def _process_and_save_tiles(self, image, label, image_name, tile_image_subdir, tile_label_subdir):
        tiles = self._extract_tiles(image, label, image_name)
        self._save_tiles(tiles, tile_image_subdir, tile_label_subdir)

    def _process_and_save_defective_tiles(self, image, label, image_name, tile_image_subdir, tile_label_subdir):
        defective_tiles = self._extract_defective_tiles(image, label, image_name)
        self._save_tiles(defective_tiles, tile_image_subdir, tile_label_subdir)

    def _save_tiles(self, tiles, tile_image_subdir, tile_label_subdir):
        for category, tile_list in tiles.items():
            for tile in tile_list:
                tile.save_tile(tile_image_subdir / category)
                tile.save_label_tile(tile_label_subdir / category)

    def process(self, data: dict, overwrite_data: bool) -> dict:
        new_data = {}
        for split, path in data.items():
            log.info(f"Tiling data in {path} ...")
            tiled_dir = self._split_images(path, overwrite_data)
            new_data[split] = tiled_dir
        return new_data