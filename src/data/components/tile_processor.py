import cv2
import os
import numpy as np
from tqdm import tqdm
from .helper_utils import (get_file_paths_rec,
                           get_file_name,
                           find_annotation_file)

from .tile_iterator import TileIterator
from .label_strategies.label_strategy_factory import get_label_strategy


class TilingProcessor(TileIterator):
    def __init__(self,
                 tile_width: int = 128,
                 tile_height: int = 128,
                 min_defective_area: float = 0.1,
                 overlap: int = 64,
                 images_dir: str = 'images',
                 labels_dir: str = 'labels',
                 tiles_dir: str = 'tiles'):
        """
        Initialize TilingProcessor with tile dimensions, minimum defective area, overlap,
        and directory names for images, labels, and tiles.
        """
        super().__init__(tile_width = tile_width,
                         tile_height = tile_height,
                         min_defective_area = min_defective_area,
                         overlap = overlap)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.tiles_dir = tiles_dir

    def process(self,
                path: str,
                use_tqdm: bool = True) -> None:
        """
        Process all images in the given directory, creating tiles and corresponding labels.
        """
        image_list = get_file_paths_rec(os.path.join(path, self.images_dir))
        label_source_dir = os.path.join(path, self.labels_dir)
        tile_images_dir = os.path.join(path, self.tiles_dir, self.images_dir)
        tile_labels_dir = os.path.join(path, self.tiles_dir, self.labels_dir)

        progress = tqdm(image_list, desc='Processing images') if use_tqdm else image_list
        for image_path in progress:
            image_name = get_file_name(image_path)
            image = cv2.imread(str(image_path))
            label = self.__get_label(image, image_name, label_source_dir)
            tiles = self._get_tiles(image, label, image_name)
            self.__save_tiles(tiles, tile_images_dir, tile_labels_dir)

    def __get_label(self,
                    image: np.array,
                    image_name: str,
                    label_source_dir: str) -> np.array:
        """
        Retrieve the label for the given image, or create a blank label if none exists.
        """
        label_path = find_annotation_file(label_source_dir, image_name)
        if label_path is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        file_extension = os.path.splitext(label_path)[-1].lower()
        label_strategy = get_label_strategy(file_extension)

        return label_strategy.process_label(label_path, image.shape)
    
    def __save_tiles(self,
                     tiles: dict,
                     tile_images_dir: str,
                     tile_labels_dir: str) -> None:
        """
        Save the generated tiles and their labels to the respective directories.
        """
        for key, tile_list in tiles.items():
            for tile in tile_list:
                tile.save_tile(os.path.join(tile_images_dir, key))
                tile.save_label_tile(os.path.join(tile_labels_dir, key))