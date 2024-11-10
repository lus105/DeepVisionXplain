from typing import Optional, Any
import torch
from pathlib import Path
import timm
import segmentation_models_pytorch as smp
import torchvision.models as models


def weight_load(
    ckpt_path: str, remove_prefix: str = "net.", ext: str = ".ckpt"
) -> dict:
    """Model weight loading helper function.

    Args:
        ckpt_path (str): Path of the weights.
        remove_prefix (str, optional): Remove prefix from keys. Defaults to "net.".
        ext (str, optional): Checkpoint extension. Defaults to ".ckpt".

    Returns:
        dict: Model weights.
    """
    if not ckpt_path.endswith(ext):
        searched_path = Path(ckpt_path)
        ckpt_path = next(searched_path.rglob("*" + ext), "")

    checkpoint = torch.load(ckpt_path)
    model_weights = {
        k[len(remove_prefix) :]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(remove_prefix)
    }

    return model_weights


def create_model(
        model_name: str,
        model_repo: Optional[str] = None,
        **kwargs: Any
) -> torch.nn.Module:
    if "torchvision.models" in model_name:
        model_name = model_name.split("torchvision.models/")[1]
        model = getattr(models, model_name)(**kwargs)
    elif "segmentation_models_pytorch" in model_name:
        model_name = model_name.split("segmentation_models_pytorch/")[1]
        model = getattr(smp, model_name)(**kwargs)
    elif "timm" in model_name:
        model_name = model_name.split("timm/")[1]
        model = timm.create_model(model_name, **kwargs)
    elif "torch.hub" in model_name:
        model_name = model_name.split("torch.hub/")[1]
        if not model_repo:
            raise ValueError("Please provide model_repo for torch.hub")
        model = torch.hub.load(model_repo, model_name, **kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")
    
    return model