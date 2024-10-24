import os
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchsampler import ImbalancedDatasetSampler

from src.data.components.data_splitter import DatasetSplitter
from src.data.components.tile_processor import TilingProcessor
from src.data.components.image_label_dataset import ImageLabelDataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DirDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_subdir: str = "train/",
        val_subdir: str = "val/",
        test_subdir: str = "test/",
        image_subdir: str = "images/",
        label_subdir: str = "labels/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        preprocessor: TilingProcessor = None,
        oversample: bool = False,
        save_predict_images: bool = False,
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
        self.data_predict: Optional[Dataset] = None

        self.data_splitter = DatasetSplitter(
            os.path.join(self.hparams.data_dir, self.hparams.image_subdir),
            os.path.join(self.hparams.data_dir, self.hparams.label_subdir),
        )

        self.updated_dirs = {}

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes (2).
        """
        return 2

    def prepare_data(self) -> None:
        """Prepare data."""

        log.info(f"Preparing data in {self.hparams.data_dir}...")
        # Define directories for train, test, and validation subsets.
        subsets = ["train", "test", "val"]
        dirs = {
            subset: os.path.join(
                self.hparams.data_dir, getattr(self.hparams, f"{subset}_subdir")
            )
            for subset in subsets
        }

        log.info("Sptillting datasets...")
        # Save data splits.
        self.data_splitter.save_splits(
            dirs["train"],
            dirs["test"],
            dirs["val"],
            self.hparams.image_subdir,
            self.hparams.label_subdir,
        )

        log.info("Preprocessing data...")
        # Preprocess data for each subset.
        self.updated_dirs = {}  # Initialize a dictionary to store updated paths
        for subset, dir in dirs.items():
            self.updated_dirs[subset] = self.hparams.preprocessor.process(dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        if not hasattr(self, "updated_dirs"):
            raise ValueError(
                "Data directories are not prepared or updated paths are missing."
            )

        transform_map = {
            "train": self.train_transforms,
            "val": self.val_test_transforms,
            "test": self.val_test_transforms,
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
            transform=self.val_test_transforms,
            label_transform=self.val_test_transforms,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._default_dataloader(
            self.data_train, shuffle=False, oversample=self.hparams.oversample
        )

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
        self, dataset: Dataset, shuffle: bool = False, oversample: bool = False
    ) -> DataLoader[Any]:
        """Create and return a dataloader.

        :param dataset: The dataset to use.
        """
        return DataLoader(
            dataset=dataset,
            sampler=ImbalancedDatasetSampler(dataset) if oversample else None,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    _ = DirDataModule()
