from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
)

from .components.nn_utils import weight_load


class SegmentationLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.modules.loss,
        compile: bool,
        ckpt_path: str,
    ) -> None:
        """Initialize lightning module.

        Args:
            net (torch.nn.Module): Model used.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
            loss (torch.nn.modules.loss): Loss function.
            compile (bool): Compile model.
            ckpt_path (string): Model chekpoint path.
        """
        super().__init__()

        # model
        self.net = net
        # optimizer
        self.optimizer = optimizer
        # scheduler
        self.scheduler = scheduler
        # loss function
        self.criterion = loss
        # compile model
        self.compile = compile
        # checkpoint path
        self.ckpt_path = ckpt_path

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # segmentation metrics collection
        seg_metrics = MetricCollection(
            {
                'accuracy': BinaryAccuracy(),
                'f1_score': BinaryF1Score(),
                'precision': BinaryPrecision(),
                'recall': BinaryRecall(),
                'iou': BinaryJaccardIndex(),
            }
        )
        self.train_metrics = seg_metrics.clone()
        self.test_metrics = seg_metrics.clone()
        self.val_metrics = seg_metrics.clone()
        # for tracking best iou score
        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        Args:
            x (torch.Tensor): A tensor of images.

        Returns:
            torch.Tensor: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.train_metrics.reset()
        self.test_metrics.reset()
        self.val_metrics.reset()
        self.val_iou_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        images, masks = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, masks)
        preds = torch.sigmoid(outputs)
        return loss, preds, masks

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metrics(preds, targets)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/iou', self.train_metrics['iou'], on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.train_metrics.items():
            if metric_name != 'iou':
                self.log(f'train/{metric_name}', metric, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_metrics(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_metrics['iou'], on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.val_metrics.items():
            if metric_name != 'iou':
                self.log(f'val/{metric_name}', metric, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        iou = self.val_metrics['iou'].compute()
        self.val_iou_best(iou)
        self.log('val/iou_best', self.val_iou_best, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_metrics['iou'], on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.test_metrics.items():
            if metric_name != 'iou':
                self.log(f'test/{metric_name}', metric, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_dict({f"test/{k}": v for k, v in self.test_metrics.compute().items()}, prog_bar=True)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single predict step on a batch of data from the test set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.
        """
        pass

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == 'fit':
            self.net = torch.compile(self.net)
        if self.ckpt_path:
            model_weights = weight_load(self.ckpt_path)
            self.net.load_state_dict(model_weights)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Returns:
            Dict[str, Any]: A dict containing the configured optimizers and
            learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}
