from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from lightly.data import LightlyDataset
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class UnsupervisedDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: Compose = None,
        val_test_transforms: Compose = None,
        num_classes: int = 2,
    ) -> None:
        """Initialize a `DirDataModule`.

        Args:
            data_dir (str, optional): The data directory. Defaults to 'data/'.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            train_transforms (Compose, optional): Train split transformations. Defaults to None.
            val_test_transforms (Compose, optional): Validation and test split transformations. Defaults to None.
            num_classes (int, optional): Number of classes in the dataset.
        """
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms
        self.val_test_transforms = val_test_transforms
        self._num_classes = num_classes

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        Returns:
            int: The number of classes.
        """
        return self._num_classes

    def prepare_data(self) -> None:
        """Data preparation hook."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Datamodule setup step.

        Args:
            stage (Optional[str], optional): The stage to setup. Either `"fit"`,
            `"validate"`, `"test"`, or `"predict"`. Defaults to None.
        """
        self.data_train = LightlyDataset(
            input_dir=self.data_dir,
            transform=self.train_transforms,
        )

        self.data_test = LightlyDataset(
            input_dir=self.data_dir,
            transform=self.val_test_transforms,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            DataLoader[Any]: The train dataloader.
        """
        return self._default_dataloader(self.data_train, shuffle=True, drop_last=True)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            DataLoader[Any]: The test dataloader.
        """
        return self._default_dataloader(self.data_test, shuffle=False, drop_last=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (Optional[str], optional): The stage being torn down. Either `"fit"`,
            `"validate"`, `"test"`, or `"predict"`. Defaults to None.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
            Dict[Any, Any]: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict (Dict[str, Any]): The datamodule state returned by `self.state_dict()`.
        """
        pass

    def _default_dataloader(
        self, dataset: Dataset, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader[Any]:
        """Create and return a dataloader.

        Args:
            dataset (Dataset): The dataset to use.
            shuffle (bool, optional): Flag for shuffling data. Defaults to False.

        Returns:
            DataLoader[Any]: Pytorch dataloader.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=shuffle,
            drop_last=drop_last
        )