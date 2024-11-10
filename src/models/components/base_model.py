from typing import Any, Optional

import timm
import torch
from torch import nn
import torchvision.models as models
import segmentation_models_pytorch as seg_models

class BaseModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Base Module Wrapper for PyTorch like model.

        Available models registries:

        - torchvision.models
        - segmentation_models_pytorch
        - timm
        - torch.hub

        Args:
            model_name (str): Model name.
            model_repo (Optional[str], optional): Model repository. Defaults to None.

        """
        super().__init__()
        if "torchvision.models" in model_name:
            model_name = model_name.split("torchvision.models/")[1]
            self.model = getattr(models, model_name)(**kwargs)
        elif "segmentation_models_pytorch" in model_name:
            model_name = model_name.split("segmentation_models_pytorch/")[1]
            self.model = getattr(seg_models, model_name)(**kwargs)
        elif "timm" in model_name:
            model_name = model_name.split("timm/")[1]
            self.model = timm.create_model(model_name, **kwargs)
        elif "torch.hub" in model_name:
            model_name = model_name.split("torch.hub/")[1]
            if not model_repo:
                raise ValueError("Please provide model_repo for torch.hub")
            self.model = torch.hub.load(model_repo, model_name, **kwargs)
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
