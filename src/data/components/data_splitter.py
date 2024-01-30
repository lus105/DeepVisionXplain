import os
from sklearn.model_selection import train_test_split

from src.data.components.helper_utils import save_files, get_file_paths_rec

class DatasetSplitter:
    def __init__(self,
                 images_path : str,
                 labels_path: str,
                 test_size: float = 0.2,
                 val_size: float =0.1,
                 random_state: int = 42):
        """
        Initialize a `DatasetSplitter`.

        :param images_path: Path to the directory containing images.
        :param labels_path: Path to the directory containing labels.
        :param test_size: Proportion of the dataset to include in the test split. Defaults to `0.2`.
        :param val_size: Proportion of the dataset to include in the validation split. Defaults to `0.1`.
        :param random_state: Controls the shuffling applied to the data before applying the split. Defaults to `42`.
        :param image_subdir: Subdirectory name for storing images. Defaults to `'images'`.
        :param label_subdir: Subdirectory name for storing labels. Defaults to `'labels'`.
        """
        self.images_path = images_path
        self.labels_path = labels_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        

    def __split_dataset(self):
        """
        Splits the dataset into training, testing, and optionally validation sets.

        :return: A tuple containing lists of image and label paths for training, testing, and validation sets. 
                 If validation size is 0, it returns lists for training and testing sets only.
        """

        images_paths = get_file_paths_rec(self.images_path)
        labels_paths = get_file_paths_rec(self.labels_path)

        # raise exeption if no images or labels found
        if len(images_paths) == 0:
            raise Exception(f'No images found in {self.images_path}')

        # raise exeption if images and labels don't match
        if len(images_paths) != len(labels_paths):
            raise Exception(f'Number of images ({len(images_paths)}) does not match number of labels ({len(labels_paths)})')
        
        # Splitting into train and test
        img_train, img_test, lbl_train, lbl_test = train_test_split(images_paths,
                                                                    labels_paths,
                                                                    test_size=self.test_size,
                                                                    random_state=self.random_state)

        # Further splitting train into train and validation
        if self.val_size > 0:
            # Adjust val size based on remaining dataset after test split
            val_size_adjusted = self.val_size / (1 - self.test_size)
            img_train, img_val, lbl_train, lbl_val = train_test_split(img_train,
                                                                      lbl_train,
                                                                      test_size=val_size_adjusted,
                                                                      random_state=self.random_state)
            
            return img_train, lbl_train, img_test, lbl_test, img_val, lbl_val
        else:
            return img_train, lbl_train, img_test, lbl_test


    def save_splits(self,
                    train_dir: str,
                    test_dir: str,
                    val_dir: str = None,
                    image_subdir: str = 'images',
                    label_subdir: str = 'labels'):
        """
        Saves the dataset splits into specified directories.

        :param train_dir: Directory to save the training set.
        :param test_dir: Directory to save the testing set.
        :param val_dir: Optional directory to save the validation set. If None, validation set is not saved.
        """
        img_train, lbl_train, img_test, lbl_test, *val_split = self.__split_dataset()

        # Define a helper function for saving files
        def save_data(image_set, label_set, directory):
            save_files(image_set, os.path.join(directory, image_subdir))
            save_files(label_set, os.path.join(directory, label_subdir))

        # Save training and testing data
        save_data(img_train, lbl_train, train_dir)
        save_data(img_test, lbl_test, test_dir)

        # Save validation data if provided
        if val_dir and val_split:
            img_val, lbl_val = val_split
            save_data(img_val, lbl_val, val_dir)