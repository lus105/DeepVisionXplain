import torch
import os
import numpy as np
import cv2
from src.utils.utils import find_file_path

def weight_load(ckpt_path: str, remove_prefix: str = "net.") -> dict:
    checkpoint_path = find_file_path(ckpt_path)
    checkpoint = torch.load(checkpoint_path)
    model_weights = {
        k[4:]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(remove_prefix)
    }

    return model_weights

def save_images(
    image: torch.Tensor, cam: torch.Tensor, label: torch.Tensor, path: str
) -> None:
    # make path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)

    cam = cam.cpu().numpy()

    label = label.cpu().numpy()
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    label = (label * 255).astype(np.uint8)

    # Thresholded cam
    cam_thresholded = np.where(cam > 0.5, 1, 0)
    cam_thresholded = (cam_thresholded * 255).astype(np.uint8)
    cam_thresholded = cv2.cvtColor(cam_thresholded, cv2.COLOR_GRAY2RGB)

    # Normalize CAM for applying colormap
    cam_normalized = cv2.normalize(
        cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    # Apply the JET colormap
    cam_colored = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)

    alpha = 0.5  # Transparency for the CAM overlay; adjust as needed
    blended_image = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)

    img_concated = cv2.hconcat(
        [image, label, cam_colored, blended_image, cam_thresholded]
    )
    cv2.imwrite(path, img_concated)