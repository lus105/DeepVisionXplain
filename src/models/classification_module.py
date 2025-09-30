from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .components.utils import weight_load


class ClassificationLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.modules.loss,
        compile: bool,
        ckpt_path: str,
        num_classes: int,
    ) -> None:
        """Initialize lightning module.

        Args:
            net (torch.nn.Module): Model used.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
            loss (torch.nn.modules.loss): Loss function.
            compile (bool): Compile model.
            ckpt_path (string): Model chekpoint path.
            num_classes (int): Number of classes.
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
        # number of classes
        self.num_classes = num_classes

        # metric objects for calculating and averaging accuracy across batches
        if self.num_classes == 2:
            self.train_acc = Accuracy(task='binary')
            self.val_acc = Accuracy(task='binary')
            self.test_acc = Accuracy(task='binary')
        else:
            self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
            self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
            self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

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

        if self.num_classes == 2 and logits.shape[-1] == 1:
            # Binary classification with single output neuron
            y = y.view(-1, 1).float()
            loss = self.criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
            y = y.squeeze()
        else:
            # Multi-class or binary with 2 output neurons
            y = y.long()
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

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
    ) -> torch.Tensor:
        """Perform a single predict step on a batch of data from the test set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple)
            containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        x, y = batch
        logits = self.forward(x)

        if self.num_classes == 2 and logits.shape[-1] == 1:
            # Binary classification with single output neuron
            preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
        else:
            # Multi-class or binary with 2 output neurons
            preds = torch.argmax(logits, dim=1)

        return preds

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.net.load_model(num_classes=self.num_classes)
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
