import os
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from src.data.components.preprocessing.preproc_pipeline_manager import PreprocessingPipeline
from src.data.components.image_label_dataset import ImageLabelDataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DirDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        preprocessing_pipeline: PreprocessingPipeline = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: Compose = None,
        val_test_transforms: Compose = None,
        save_predict_images: bool = False,
    ) -> None:
        """Initialize a `DirDataModule`.

        Args:
            data_dir (str, optional): The data directory. Defaults to "data/".
            split_ratio (list, optional): Dataset split ratio (train, test, val).
            random_state (int, optional): Data splitting random state.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            save_predict_images (bool, optional): Save images in predict mode? Defaults to False.
            train_transforms (Compose, optional): Train split transformations.
            val_test_transform (Compose, optional): Validation and test split transformations.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        Returns:
            int: The number of classes (2).
        """
        return 2

    def prepare_data(self) -> None:
        """Prepare data."""

        log.info(f"Preparing data in {self.hparams.data_dir}...")

        initial_data = {'data': self.hparams.data_dir}
        final_data = self.hparams.preprocessing_pipeline.run(initial_data)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        if not hasattr(self, "updated_dirs"):
            raise ValueError(
                "Data directories are not prepared or updated paths are missing."
            )

        transform_map = {
            "train": self.hparams.train_transforms,
            "val": self.hparams.val_test_transforms,
            "test": self.hparams.val_test_transforms,
        }

        for subset in ["train", "val", "test"]:
            dir_path = self.updated_dirs[subset]
            # For ImageFolder, there's no separate label_dir
            self.__dict__[f"data_{subset}"] = ImageFolder(
                root=dir_path,
                transform=transform_map[subset],
            )

        # For ImageLabelDataset, specify img_dir and label_dir
        self.data_predict = ImageLabelDataset(
            img_dir=self.updated_dirs["test"],
            label_dir=os.path.join(
                os.path.dirname(self.updated_dirs["test"]), self.hparams.label_subdir
            ),
            transform=self.hparams.val_test_transforms,
            label_transform=self.hparams.val_test_transforms,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._default_dataloader(self.data_train, shuffle=False)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._default_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._default_dataloader(self.data_test, shuffle=False)
    
    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return self._default_dataloader(self.data_predict, shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def _default_dataloader(
        self, dataset: Dataset, shuffle: bool = False) -> DataLoader[Any]:
        """Create and return a dataloader.

        :param dataset: The dataset to use.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )
