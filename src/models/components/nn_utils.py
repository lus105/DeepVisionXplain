import torch
import os
import numpy as np
import cv2
from pathlib import Path
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
    opset_version: int = 20,
) -> None:
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


def save_images(
    image: torch.Tensor, map: torch.Tensor, label: torch.Tensor, path: str
) -> None:
    """Image saving function for image, explainability map and label. Images are
    horizontally concatenated.

    Args:
        image (torch.Tensor): Original image.
        map (torch.Tensor): Explainability map.
        label (torch.Tensor): True label (mask).
        path (str): Path for image saving.
    """
    # make path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)

    map = map.cpu().numpy()

    label = label.cpu().numpy()
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    label = (label * 255).astype(np.uint8)

    # Thresholded map
    map_thresholded = np.where(map > 0.5, 1, 0)
    map_thresholded = (map_thresholded * 255).astype(np.uint8)
    map_thresholded = cv2.cvtColor(map_thresholded, cv2.COLOR_GRAY2RGB)

    # Normalize map for applying colormap
    map_normalized = cv2.normalize(
        map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    # Apply the JET colormap
    map_colored = cv2.applyColorMap(map_normalized, cv2.COLORMAP_JET)

    alpha = 0.5  # Transparency for the map overlay
    blended_image = cv2.addWeighted(image, 1 - alpha, map_colored, alpha, 0)

    img_concated = cv2.hconcat(
        [image, label, map_colored, blended_image, map_thresholded]
    )
    cv2.imwrite(path, img_concated)
