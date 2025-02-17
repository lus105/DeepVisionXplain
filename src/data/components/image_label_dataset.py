from pathlib import Path
from typing import Callable, Optional
import cv2
import torch
from torch.utils.data import Dataset
import albumentations
import torchvision
from PIL import Image
from .utils import list_files, IMAGE_EXTENSIONS


class ImageLabelDataset(Dataset):
    """
    A dataset class for loading image-label pairs from directories and their subdirectories.
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        label_postfix: str = '',
    ) -> None:
        """Initialize the dataset.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing labels.
            transform (Optional[Callable], optional): Transform applied
              to image-label pairs. Defaults to None.
            label_postfix (str): image name label postfix. Defaults to ''
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.label_postfix = label_postfix
        self.img_label_pairs = self._get_img_label_pairs()

    def _get_img_label_pairs(self) -> list[tuple[Path, Path]]:
        """Find valid image-label pairs.

        Returns:
            list[tuple[Path, Path]]: List of (image_path, label_path) pairs.
        """
        # Get all image and label files
        img_files = list_files(self.img_dir, IMAGE_EXTENSIONS)
        label_files = list_files(self.label_dir, IMAGE_EXTENSIONS)

        # Create a lookup dictionary for label files by stem name (without "_label" suffix)
        label_lookup = {
            label_file.stem: label_file
            for label_file in label_files
        }

        # Match image files with corresponding label files
        img_label_pairs = []
        for img_path in img_files:
            # Construct the expected label stem
            expected_label_stem = f"{img_path.stem}{self.label_postfix}"
            label_path = label_lookup.get(expected_label_stem)
            if label_path and label_path.exists():
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            # Handle Albumentations transforms
            if isinstance(self.transform, albumentations.core.composition.BaseCompose):
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']
                label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0) / 255.0

            # Handle TorchVision transforms
            elif isinstance(self.transform, torchvision.transforms.Compose):
                image = self.transform(
                    Image.fromarray(image)
                )  # Convert image to PIL for torchvision
                label = torch.tensor(label, dtype=torch.float32) / 255.0
            else:
                raise ValueError(
                    'Unsupported transform type. Use Albumentations or TorchVision transforms.'
                )
            
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, label