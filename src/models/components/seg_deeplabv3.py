import torch
import torch.nn as nn
from .base_model import get_model
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DeepLabV3(nn.Module):
    def __init__(
        self,
        model_name: str ='torch.hub/deeplabv3_resnet50',
        model_repo: str = 'pytorch/vision:v0.10.0',
        pretrained: bool = True,
        num_classes: int = 1,
    ) -> None:
        """Initialize DeepLabV3 model

        Args:
            model_name (str, optional): Model to load. Defaults to 'torch.hub/deeplabv3_resnet50'.
            model_repo (str, optional): Model repository. Defaults to 'pytorch/vision:v0.10.0'.
            pretrained (bool, optional): Use pretrained model. Defaults to 'IMAGENET1K_V1'.
            num_classes (int, optional): Number of classes. Defaults to 1.
        """
        super().__init__()
        self.num_classes = num_classes
        self.model = get_model(model_name, model_repo, pretrained=pretrained)
        self._modify_model()
        log.info("DeepLabV3 model created.")

    def _modify_model(self) -> None:
        """Modifies model for custom num_classes
        """
        self.model.classifier[4] = nn.Conv2d(256, self.num_classes, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)['out']
        return x