import numpy as np
import cv2
from PIL import Image
from IPython.display import display


def apply_cm(
    image: np.array, map: np.array, threshold: float = 0.5, alpha: float = 0.5
) -> np.array:
    """Apply color map on image

    Args:
        image (np.array): Image
        map (np.array): Grayscale, unnormalized cm.
        threshold (float, optional): Threshold for map. Defaults to 0.5.
        alpha (float, optional): Transparency. Defaults to 0.5.

    Returns:
        np.array: Image with applied cm.
    """
    map_thresholded = np.where(map > threshold, 1, 0)
    map_thresholded = (map_thresholded * 255).astype(np.uint8)
    map_thresholded = cv2.cvtColor(map_thresholded, cv2.COLOR_GRAY2RGB)

    map_normalized = cv2.normalize(
        map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    map_colored = cv2.applyColorMap(map_normalized, cv2.COLORMAP_JET)
    blended_image = cv2.addWeighted(image, 1 - alpha, map_colored, alpha, 0)

    return blended_image

def display_img_with_map(out: np.array, map: np.array, image: np.array) -> None:
    """Displays image with explainability map.

    Args:
        out (np.array): Neural network prediction.
        map (np.array): Neural network explainability output.
        image (np.array): Original inference image.
    """
    preds = out > 0.5
    for i, pred in enumerate(preds):
        map_segmentation = map[i] if pred == 1 else np.zeros_like(map[i])
        if pred == 1:
            map_segmentation = (map_segmentation - map_segmentation.min()) / (
                map_segmentation.max() - map_segmentation.min()
            )
        blended_img = apply_cm(image, map_segmentation)

        # Only for displaying
        blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
        display(Image.fromarray(blended_img))