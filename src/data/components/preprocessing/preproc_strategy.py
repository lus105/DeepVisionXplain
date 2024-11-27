from abc import ABC, abstractmethod

class PreprocessingStep(ABC):
    def __init__(self,
                 train_subdir: str = 'train',
                 test_subdir: str = 'test',
                 val_subdir: str = 'val',
                 image_subdir: str = 'images',
                 label_subdir: str = 'labels'):
        """
        Abstract base class for image preprocessing.

        Args:
            train_subdir (str, optional): Train subdirectory name. Defaults to 'train'.
            test_subdir (str, optional): Test subdirectory name. Defaults to 'test'.
            val_subdir (str, optional): Validation subdirectory name. Defaults to 'val'.
            image_subdir (str, optional): Image subdirectory name. Defaults to 'images'.
            label_subdir (str, optional): Label subdirectory name. Defaults to 'labels'.
        """
        self._train_subdir = train_subdir
        self._test_subdir = test_subdir
        self._val_subdir = val_subdir
        self._image_subdir = image_subdir
        self._label_subdir = label_subdir

    @abstractmethod
    def process(self, data: dict) -> dict:
        """
        Process the data and return the modified data directories.
        
        Args:
            data (dict): Input data dictionary containing input directories.
        Returns:
            dict: Modified data dictionary of directories after processing.
        """
        pass
    
    @abstractmethod
    def get_processed_data_path(self, data: dict) -> dict:
        """
        Get only path of preprocessed data
        
        Args:
            data (dict): Input data dictionary containing input directories.
        Returns:
            dict: Modified data dictionary of directories after processing.
        """
        pass