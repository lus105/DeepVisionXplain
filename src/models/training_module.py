from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex)
from src.models.components.metrics import PointingGameAccuracy

from src.utils import save_images

class TrainingLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        segmentation_test: bool = False,
        save_images: bool = False,
    ) -> None:
        """Initialize a `CnnLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.BCELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary", threshold=0.5)
        self.val_acc = Accuracy(task="binary", threshold=0.5)
        self.test_acc = Accuracy(task="binary", threshold=0.5)
        self.seg_bin_acc = BinaryAccuracy(threshold=0.5)
        self.seg_bin_f1 = BinaryF1Score(threshold=0.5)
        self.seg_bin_precision = BinaryPrecision(threshold=0.5)
        self.seg_bin_recall = BinaryRecall(threshold=0.5)
        self.seg_bin_jaccard = BinaryJaccardIndex(threshold=0.5)
        self.pointing_game_acc = PointingGameAccuracy()
        self.seg_metrics = [self.seg_bin_acc,
                            self.seg_bin_f1,
                            self.seg_bin_precision,
                            self.seg_bin_recall,
                            self.seg_bin_jaccard,
                            self.pointing_game_acc]
        self.save_images = save_images
        self.counter = 0

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
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
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        y = y.view(-1, 1).float()
        logits = torch.sigmoid(self.forward(x))
        loss = self.criterion(logits, y)
        preds = (logits > 0.5).float()
        return loss, preds, y
    
    def model_step_segmentation_test(
        self, batch
    ) -> None:
        x, y = batch
        out, cam = self.forward(x)
        logit_out = torch.sigmoid(out)
        preds = (logit_out > 0.5).float()

        for i, pred in enumerate(preds):
            if pred == 1:
                # Use cam[i] for segmentation metric calculation if pred is 1
                cam_segmentation = cam[i]
                # convert cam values to range [0, 1]
                cam_segmentation = (cam_segmentation - cam_segmentation.min()) / (cam_segmentation.max() - cam_segmentation.min())

                # save images for visualization
                if self.save_images:
                    save_images(x[i], cam_segmentation, y[i].squeeze(0), f"{self._trainer.default_root_dir}/images/img_{self.counter}.png")

                # thresholding cam_segmentation to get binary mask
                cam_segmentation = (cam_segmentation > 0.5).float()

                # Placeholder function to calculate segmentation metric, implement accordingly
                for metric in self.seg_metrics:
                    metric(cam_segmentation, y[i].squeeze(0))

                self.counter += 1
            else:
                # Use a blank image for segmentation metric calculation if pred is 0
                blank_image = torch.zeros_like(cam[i])
                for metric in self.seg_metrics:
                    metric(blank_image, y[i].squeeze(0))
                self.counter += 1

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if self.hparams.segmentation_test:
            loss = self.model_step_segmentation_test(batch)
        else:
            loss, preds, targets = self.model_step(batch)
            # update and log metrics
            self.test_loss(loss)
            self.test_acc(preds, targets)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if self.hparams.segmentation_test:
            for metric in self.seg_metrics:
                self.log(f"test/{metric._get_name()}", metric.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = TrainingLitModule(None, None, None, None)
