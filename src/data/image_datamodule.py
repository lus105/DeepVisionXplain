from pathlib import Path
from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from src.data.components.preprocessing.preproc_pipeline_manager import (
    PreprocessingPipeline,
)
from src.data.components.image_label_dataset import ImageLabelDataset
from src.data.components.utils import clear_directory
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/',
        preprocessing_pipeline: PreprocessingPipeline = None,
        overwrite_data: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: Compose = None,
        val_test_transforms: Compose = None,
        save_predict_images: bool = False,
        num_classes: int = 2,
    ) -> None:
        """Initialize a `DirDataModule`.

        Args:
            data_dir (str, optional): The data directory. Defaults to 'data/'.
            preprocessing_pipeline (PreprocessingPipeline, optional): Custom preprocessing pipeline. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            train_transforms (Compose, optional): Train split transformations. Defaults to None.
            val_test_transforms (Compose, optional): Validation and test split transformations. Defaults to None.
            save_predict_images (bool, optional): Save images in predict mode? Defaults to False.
            num_classes (int, optional): Number of classes in the dataset.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        self.preprocessed_data: dict[Path] = {}

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        Returns:
            int: The number of classes (2).
        """
        return self.hparams.num_classes

    def prepare_data(self) -> None:
        """Data preparation hook."""
        log.info(f'Preparing data in {self.hparams.data_dir}...')

        data_path = Path(self.hparams.data_dir)
        base_path = data_path.parent
        last_subdir = data_path.name
        output_path = base_path / f'{last_subdir}_processed'

        initial_data = {'initial_data': self.hparams.data_dir}
        if output_path.exists():
            if self.hparams.overwrite_data:
                clear_directory(output_path)
                output_path.rmdir()
                self.preprocessed_data = self.hparams.preprocessing_pipeline.run(
                    initial_data
                )
            else:
                self.preprocessed_data = (
                    self.hparams.preprocessing_pipeline.get_processed_data_path(
                        initial_data
                    )
                )
        else:
            self.preprocessed_data = self.hparams.preprocessing_pipeline.run(
                initial_data
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Datamodule setup step.

        Args:
            stage (Optional[str], optional): The stage to setup. Either `"fit"`,
            `"validate"`, `"test"`, or `"predict"`. Defaults to None.
        """
        self.data_train = ImageFolder(
            root=self.preprocessed_data['train'],
            transform=self.hparams.train_transforms,
        )

        self.data_test = ImageFolder(
            root=self.preprocessed_data['test'],
            transform=self.hparams.val_test_transforms,
        )

        self.data_val = ImageFolder(
            root=self.preprocessed_data['val'],
            transform=self.hparams.val_test_transforms,
        )

        self.data_predict = ImageLabelDataset(
            img_dir=self.preprocessed_data['test'],
            label_dir=self.preprocessed_data['test'].parent / 'labels',
            transform=self.hparams.val_test_transforms,
            label_transform=self.hparams.val_test_transforms,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            DataLoader[Any]: The train dataloader.
        """
        return self._default_dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader[Any]: The validation dataloader.
        """
        return self._default_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            DataLoader[Any]: The test dataloader.
        """
        return self._default_dataloader(self.data_test, shuffle=False)

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        Returns:
            DataLoader[Any]: The predict dataloader.
        """
        return self._default_dataloader(self.data_predict, shuffle=False)

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
        self, dataset: Dataset, shuffle: bool = False
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
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )
