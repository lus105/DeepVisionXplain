import torch
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from scipy.ndimage import label, center_of_mass

class PointingGameAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Ensure input and target have the same dimensions
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        # Check if both images are completely blank (only contain zeros)
        if torch.all(preds == 0) and torch.all(target == 0):
            return  # Skip incrementing correct and total if both are blank

        # Convert tensors to numpy for processing with scipy
        preds_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()

        # Label connected components in the prediction.
        structure = np.ones((3, 3), dtype=np.int32)  # Define connectivity
        labeled_array, num_features = label(preds_np, structure=structure)
        
        # Skip if no white regions are detected
        if num_features == 0:
            self.total += 1
            return

        # Find the largest connected component
        largest_component = max(range(1, num_features + 1), key=lambda x: (labeled_array == x).sum())

        # Compute the centroid of the largest component
        centroid = center_of_mass(preds_np, labels=labeled_array, index=largest_component)
        centroid = tuple(int(x) for x in centroid)  # Convert to integer indices

        # Increment total predictions count
        self.total += 1

        # Check if the centroid is correctly labeled in the target
        if target_np[centroid] == 1:
            self.correct += 1

    def compute(self) -> Tensor:
        # Calculate accuracy
        if self.total == 0:
            return torch.tensor(0.)
        return self.correct.float() / self.total