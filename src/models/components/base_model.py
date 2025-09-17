from typing import Any, Optional

import timm
import torch
from torch import nn
import torchvision.models as models
import segmentation_models_pytorch as seg_models


def get_model(
    model_name: str, model_repo: Optional[str] = None, **kwargs: Any
) -> nn.Module:
    """Available models registries:

        - torchvision.models
        - segmentation_models_pytorch
        - timm
        - torch.hub

    Args:
        model_name (str): Model name.
        model_repo (Optional[str], optional): Model repository. Defaults to None.

    Returns:
        nn.Module: PyTorch like model
    """
    if 'torchvision.models' in model_name:
        model_name = model_name.split('torchvision.models/')[1]
        model = getattr(models, model_name)(**kwargs)
    elif 'segmentation_models_pytorch' in model_name:
        model_name = model_name.split('segmentation_models_pytorch/')[1]
        model = getattr(seg_models, model_name)(**kwargs)
    elif 'timm' in model_name:
        model_name = model_name.split('timm/')[1]
        model = timm.create_model(model_name, **kwargs)
    elif 'torch.hub' in model_name:
        model_name = model_name.split('torch.hub/')[1]
        if not model_repo:
            raise ValueError('Please provide model_repo for torch.hub')
        model = torch.hub.load(model_repo, model_name, **kwargs)
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented')

    return model


class BaseModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Base Module Wrapper for PyTorch like model.

        Args:
            model_name (str): Model name.
            model_repo (Optional[str], optional): Model repository. Defaults to None.

        """
        super().__init__()
        self.model_name = model_name
        self.model_repo = model_repo
        self.model_kwargs = kwargs
        self.model = None
    
    def load_model(self, **additional_kwargs: Any) -> None:
        """Load or reload the model with saved kwargs and optional additional kwargs.
        
        Args:
            **additional_kwargs: Additional keyword arguments to override or extend saved kwargs.
        """
        merged_kwargs = {**self.model_kwargs, **additional_kwargs}
        self.model = get_model(self.model_name, self.model_repo, **merged_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
