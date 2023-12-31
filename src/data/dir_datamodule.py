import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class DirDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_subdir: str = "train/",
        val_subdir: str = "val/",
        test_subdir: str = "test/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DirDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_dir: Train subdirectory. Defaults to `"train/"`.
        :param val_dir: Val subdirectory. Defaults to `"val/"`.
        :param test_dir: Test subdirectory. Defaults to `"test/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.val_test_transforms = transforms.Compose([transforms.ToTensor()])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes (2).
        """
        return 2

    def prepare_data(self) -> None:
        # Assuming the directory dataset doesn't need downloading.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """

        # Check if the train and test subdirectories exist
        train_dir_exists = os.path.exists(
            os.path.join(self.hparams.data_dir, self.hparams.train_subdir)
        )
        test_dir_exists = os.path.exists(
            os.path.join(self.hparams.data_dir, self.hparams.test_subdir)
        )
        val_dir_exists = os.path.exists(
            os.path.join(self.hparams.data_dir, self.hparams.val_subdir)
        )

        # If they exist, set train and test datasets directly without splitting
        if train_dir_exists and test_dir_exists and val_dir_exists:
            self.data_train = ImageFolder(
                os.path.join(self.hparams.data_dir, self.hparams.train_subdir),
                transform=self.train_transforms,
            )
            self.data_test = ImageFolder(
                os.path.join(self.hparams.data_dir, self.hparams.test_subdir),
                transform=self.train_transforms,
            )
            self.data_val = ImageFolder(
                os.path.join(self.hparams.data_dir, self.hparams.val_subdir),
                transform=self.val_test_transforms,
            )
        else:
            raise ValueError("Train, test or val subdirectories do not exist.")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._default_dataloader(self.data_train, shuffle=True)

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

    def _default_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader[Any]:
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


if __name__ == "__main__":
    _ = DirDataModule()