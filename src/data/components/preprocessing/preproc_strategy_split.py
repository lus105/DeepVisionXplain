from pathlib import Path
from sklearn.model_selection import train_test_split
from .preproc_strategy import PreprocessingStep
from ..utils import (save_files,
                            get_file_paths_rec,
                            IMAGE_EXTENSIONS,
                            XML_EXTENSION,
                            JSON_EXTENSION)

class SplitStep(PreprocessingStep):
    def __init__(
        self,
        split_ratio,
        random_state=42,
        train_subdir="train",
        test_subdir="test",
        val_subdir="val",
        image_subdir="images",
        label_subdir="labels",
    ):
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.train_subdir = train_subdir
        self.test_subdir = test_subdir
        self.val_subdir = val_subdir
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir

    def process(self, data: dict) -> dict:
        if len(data) != 1:
            raise ValueError("Data must contain exactly one element")

        train_size, test_size, val_size = self.split_ratio
        total = sum(self.split_ratio)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("split_ratio must sum to 1.0")

        data_path = Path(next(iter(data.values())))
        image_path = data_path / self.image_subdir
        label_path = data_path / self.label_subdir

        train_path = data_path / self.train_subdir
        test_path = data_path / self.test_subdir
        val_path = data_path / self.val_subdir

        images_paths = get_file_paths_rec(image_path, file_extensions=IMAGE_EXTENSIONS)
        labels_paths = get_file_paths_rec(label_path, file_extensions=[XML_EXTENSION, JSON_EXTENSION])

        if not images_paths:
            raise FileNotFoundError(f"No images found in {image_path}")

        if len(images_paths) != len(labels_paths):
            raise ValueError(
                f"Number of images ({len(images_paths)}) does not match number of labels ({len(labels_paths)})"
            )
        
        # Splitting into train and test
        img_train, img_test, lbl_train, lbl_test = train_test_split(
            images_paths,
            labels_paths,
            test_size=test_size,
            random_state=self.random_state,
        )

        # Further splitting train into train and validation
        if val_size > 0:
            # Adjust val size based on remaining dataset after test split
            val_size_adjusted = val_size / (1 - test_size)
            img_train, img_val, lbl_train, lbl_val = train_test_split(
                img_train,
                lbl_train,
                test_size=val_size_adjusted,
                random_state=self.random_state,
            )
        
        save_files(img_train, train_path / self.image_subdir)
        save_files(lbl_train, train_path / self.label_subdir)

        save_files(img_test, test_path / self.image_subdir)
        save_files(lbl_test, test_path / self.label_subdir)

        if img_val and lbl_val:
            save_files(img_test, val_path / self.image_subdir)
            save_files(lbl_test, val_path / self.label_subdir)

        data['split_dirs'] = {
        'train': data_path / self.train_subdir,
        'test': data_path / self.test_subdir,
        'val': data_path / self.val_subdir,
        }

        return data