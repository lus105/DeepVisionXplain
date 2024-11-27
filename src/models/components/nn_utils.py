import torch
from pathlib import Path


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
        (k[len(remove_prefix):] if k.startswith(remove_prefix) else k): v
        for k, v in checkpoint["state_dict"].items()
    }

    return model_weights