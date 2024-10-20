from pathlib import Path
from sklearn.model_selection import train_test_split

from .helper_utils import (save_files,
                           get_file_paths_rec,
                           IMAGE_EXTENSIONS,
                           XML_EXTENSION,
                           JSON_EXTENSION)


def split_dataset(data_path: str,
                  split_ratio: list[float, float, float],
                  random_state: int = 42,
                  train_subdir: str = "train",
                  test_subdir: str = "test",
                  val_subdir: str = "val",
                  image_subdir: str = "images",
                  label_subdir: str = "labels",
                  ) -> None:
    """Splits a dataset into training, testing, and validation sets and saves them
    into respective directories.

    Args:
        data_path (str): Path to the root dataset directory.
        split_ratio (list of float): Ratios for [train, test, val]. Must sum to 1.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.
        train_subdir (str, optional): Directory name for training data. Defaults to "train".
        test_subdir (str, optional): Directory name for testing data. Defaults to "test".
        val_subdir (str, optional): Directory name for validation data. Defaults to "val".
        image_subdir (str, optional): Subdirectory name where images are stored. Defaults to "images".
        label_subdir (str, optional): Subdirectory name where labels are stored. Defaults to "labels".

    Raises:
        ValueError: If `split_ratio` does doesn't sum to 1.
        FileNotFoundError: If no images are found in the specified image directory.
        ValueError: If the number of images and labels do not match.
    """
    train_size, test_size, val_size = split_ratio

    total = sum(split_ratio)
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("split_ratio must sum to 1.0")

    data_path = Path(data_path)
    image_path = data_path / image_subdir
    label_path = data_path / label_subdir

    train_path = data_path / train_subdir
    test_path = data_path / test_subdir
    val_path = data_path / val_subdir

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
        random_state=random_state,
    )

    # Further splitting train into train and validation
    if val_size > 0:
        # Adjust val size based on remaining dataset after test split
        val_size_adjusted = val_size / (1 - test_size)
        img_train, img_val, lbl_train, lbl_val = train_test_split(
            img_train,
            lbl_train,
            test_size=val_size_adjusted,
            random_state=random_state,
        )
    
    save_files(img_train, train_path / image_subdir)
    save_files(lbl_train, train_path / label_subdir)

    save_files(img_test, test_path / image_subdir)
    save_files(lbl_test, test_path / label_subdir)

    if img_val and lbl_val:
        save_files(img_test, val_path / image_subdir)
        save_files(lbl_test, val_path / label_subdir)
