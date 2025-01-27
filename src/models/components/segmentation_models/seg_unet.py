import torch
import torch.nn as nn
from .base_model import get_model
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class Unet(nn.Module):
    def __init__(
            self,
            model_name: str = 'segmentation_models_pytorch/Unet',
            model_repo: str = 'segmentation_models_pytorch',
            encoder_name: str = 'resnet50',
            encoder_weigths: str = 'imagenet',
            encoder_depth: int = 3,
            decoder_channels=(256, 128, 64),
            num_classes: int = 1,
    ) -> None:
        """Initialize Segformer model

        Args:
            model_name (str, optional): Model to load. Defaults to 'segmentation_models_pytorch/Unet'.
            model_repo (str, optional): Model repository. Defaults to 'segmentation_models_pytorch'.
            pretrained (bool, optional): Use pretrained model. Defaults to 'imagenet'.
            num_classes (int, optional): Number of classes. Defaults to 1.
        """
        super().__init__()
        self.model = get_model(model_name, model_repo, encoder_name=encoder_name, encoder_weigths=encoder_weigths,
                               classes=num_classes, encoder_depth=encoder_depth, decoder_channels=decoder_channels)
        log.info("Unet model created.")
        print(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
