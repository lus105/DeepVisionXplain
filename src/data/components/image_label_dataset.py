from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ImageLabelDataset(Dataset):
    """
    A dataset class for loading image-label pairs from directories and their subdirectories.
    Labels have '_mask' appended before the '.png' extension.
    """

    def __init__(self, img_dir, label_dir, transform=None, label_transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.label_transform = label_transform
        self.img_label_pairs = self._get_img_label_pairs()

    def _get_img_label_pairs(self):
        img_files = list(self.img_dir.rglob("*.*"))  # Adjust pattern if necessary
        img_label_pairs = []

        for img_path in img_files:
            relative_path = img_path.relative_to(self.img_dir)
            # Append '_mask' before '.png' in the label path
            label_path = self.label_dir / relative_path
            label_path = label_path.with_name(
                label_path.stem + "_mask" + label_path.suffix
            )
            if label_path.exists():  # Ensure the modified label path exists
                img_label_pairs.append((img_path, label_path))

        return img_label_pairs

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.img_label_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert(
            "L"
        )  # Adjust if your labels are in a different format

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return image, label
