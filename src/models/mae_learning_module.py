from typing import Any

import torch
from torch import nn
from lightning import LightningModule

class MaeLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.modules.loss,
        compile: bool,
        ckpt_path: str,
    ) -> None:
        """Initialize lightning module.

        Args:
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

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

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
        views = batch[0]
        images = views[0]  # views contains only a single view
        outputs = self.net(images)
        loss = outputs[0]
        
        self.log("train/mae_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == 'fit':
            self.net = torch.compile(self.net)
        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, weights_only=False)
            model_weights = checkpoint["model"]  # Extract the actual model state dict
            self.net.load_state_dict(model_weights, strict=False)

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