from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
)

from .components.nn_utils import weight_load, save_images


class ClassificationLitModule(LightningModule):
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

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # model
        self.net = net
        # loss function
        self.criterion = loss

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # segmentation metrics collection
        self.seg_metrics = MetricCollection(
            {
                'accuracy': BinaryAccuracy(),
                'f1_score': BinaryF1Score(),
                'precision': BinaryPrecision(),
                'recall': BinaryRecall(),
                'jaccard': BinaryJaccardIndex(),
            }
        )

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
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

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
        x, y = batch
        logits = self.forward(x)
        y = y.view(-1, 1).float()
        loss = self.criterion(logits, y)
        preds = (logits > 0.5).float()
        return loss, preds, y

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
        self.train_acc(preds, targets)
        self.log(
            'train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
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
        self.val_acc(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            'val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

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
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            'test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log('test/acc', self.test_acc.compute(), sync_dist=True, prog_bar=True)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single predict step on a batch of data from the test set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        out, map = self.forward(x)
        preds = (out > 0.5).float()

        for i, pred in enumerate(preds):
            image = x[i]
            label = y[i].squeeze(0)
            map_segmentation = map[i] if pred == 1 else torch.zeros_like(map[i])
            if pred == 1:
                map_segmentation = (map_segmentation - map_segmentation.min()) / (
                    map_segmentation.max() - map_segmentation.min()
                )
                if self.trainer.datamodule.hparams.save_predict_images:
                    filename = f'img_batch_{batch_idx}_sample_{i}.png'
                    save_images(
                        image,
                        map_segmentation,
                        label,
                        f'{self.trainer.default_root_dir}/images/{filename}',
                    )

            map_segmentation = (map_segmentation > 0.5).float()
            self.seg_metrics(map_segmentation, label)

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        metrics_dict = self.seg_metrics.compute()
        for metric_name, metric_value in metrics_dict.items():
            print(f'test/{metric_name}: {metric_value}')

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == 'fit':
            self.net = torch.compile(self.net)
        if self.hparams.ckpt_path:
            model_weights = weight_load(self.hparams.ckpt_path)
            self.net.load_state_dict(model_weights)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Returns:
            Dict[str, Any]: A dict containing the configured optimizers and
            learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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
