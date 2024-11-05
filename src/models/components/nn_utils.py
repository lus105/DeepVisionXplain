import torch
import os
import numpy as np
import cv2
from src.utils import find_file_path

def weight_load(ckpt_path: str, remove_prefix: str = "net.") -> dict:
    """Model weight loading helper function.

    Args:
        ckpt_path (str): Path of the weights.
        remove_prefix (str, optional): Remove prefix from keys. Defaults to "net.".

    Returns:
        dict: Model weights
    """
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = find_file_path(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model_weights = {
        k[len(remove_prefix):]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(remove_prefix)
    }

    return model_weights