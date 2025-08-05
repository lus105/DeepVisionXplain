import json
from pathlib import Path
import torch
from torch import nn


def weight_load(
    ckpt_path: str,
    remove_prefix: str = 'net.',
    ext: str = '.ckpt',
    weights_only: bool = False,
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
        ckpt_path = next(searched_path.rglob('*' + ext), '')

    checkpoint = torch.load(ckpt_path, weights_only=weights_only)
    model_weights = {
        (k[len(remove_prefix) :] if k.startswith(remove_prefix) else k): v
        for k, v in checkpoint['state_dict'].items()
    }

    return model_weights


def export_model_to_onnx(
    model: nn.Module,
    onnx_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    input_names: list = ['input'],
    output_names: list = ['output'],
    opset_version: int = 19,
) -> None:
    """
    Exports a PyTorch model to the ONNX format, with optional class names metadata.
    Args:
        model (nn.Module): The PyTorch model to export.
        onnx_path (str): The file path where the ONNX model will be saved.
        input_shape (tuple, optional): The shape of the dummy input tensor for tracing. Defaults to (1, 3, 224, 224).
        input_names (list, optional): Names for the model's input nodes. Defaults to ['input'].
        output_names (list, optional): Names for the model's output nodes. Defaults to ['output'].
        opset_version (int, optional): The ONNX opset version to use. Defaults to 19.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        opset_version=opset_version,
    )
