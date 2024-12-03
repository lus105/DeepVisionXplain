from pathlib import Path
import cv2
from torch.utils.data import Dataset
from .utils import find_annotation_file

class ImageLabelDataset(Dataset):
    """
    A dataset class for loading image-label pairs from directories and their subdirectories.
    Labels have '_mask' appended before the '.png' extension.
    """

    def __init__(self, img_dir, label_dir, transform=None, label_postfix=''):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.label_postfix= label_postfix
        self.img_label_pairs = self._get_img_label_pairs()

    def _get_img_label_pairs(self):
        img_files = list(self.img_dir.rglob('*.*'))
        img_label_pairs = []

        for img_path in img_files:
            label_path = find_annotation_file(self.label_dir, img_path.stem)
            if label_path.exists():
                img_label_pairs.append((img_path, label_path))

        return img_label_pairs

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.img_label_pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        return image, label
