from typing import Any

import torch
import torchvision
from lightning import LightningModule

from lightly.models.modules.heads import SimCLRProjectionHead


class ContrastiveLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.modules.loss,
        compile: bool,
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
        self.backbone = None
        self.projection_head = None
        # optimizer
        self.optimizer = optimizer
        # scheduler
        self.scheduler = scheduler
        # loss function
        self.criterion = loss
        # compile model
        self.compile = compile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): A tensor of images.

        Returns:
            torch.Tensor: A tensor of logits.
        """
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

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
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train/loss_ssl", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        resnet = torchvision.models.resnet18()
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

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
