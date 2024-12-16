from pathlib import Path
from typing import Callable, Optional
import cv2
import torch
from torch.utils.data import Dataset
from .utils import find_file_by_name, list_files, IMAGE_EXTENSIONS

class ImageLabelDataset(Dataset):
    """
    A dataset class for loading image-label pairs from directories and their subdirectories.
    """
    def __init__(self,
                 img_dir: str,
                 label_dir: str,
                 transform: Optional[Callable] = None,
                 label_postfix: str = '') -> None:
        """Initialize the dataset.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing labels.
            transform (Optional[Callable], optional): Transform applied
              to image-label pairs. Defaults to None.
            label_postfix (str, optional): Postfix for matching labels
              to images. Defaults to ''.
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.label_postfix= label_postfix
        self.img_label_pairs = self._get_img_label_pairs()

    def _get_img_label_pairs(self) -> list[tuple[Path, Path]]:
        """Find valid image-label pairs.

        Returns:
            list[tuple[Path, Path]]: List of (image_path, label_path) pairs.
        """
        img_files = list_files(self.img_dir, IMAGE_EXTENSIONS)
        img_label_pairs = []

        for img_path in img_files:
            label_path = find_file_by_name(self.label_dir, img_path.stem)
            if label_path.exists():
                img_label_pairs.append((img_path, label_path))

        return img_label_pairs

    def __len__(self) -> int:
        """Get the number of image-label pairs.

        Returns:
            int: Number of pairs.
        """
        return len(self.img_label_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an image-label pair by index.

        Args:
            idx (int): Index of the pair.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed image and label tensors.
        """
        img_path, label_path = self.img_label_pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
            label = label.to(torch.float32) / 255.0

        return image, label
