import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense detection tasks, such as RetinaNet.

    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: 2.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'. Default: 'none'.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs (Tensor): Probabilities (sigmoid already applied) for each example.
            targets (Tensor): Binary classification labels (0 or 1) with the same shape as inputs.

        Returns:
            Tensor: Loss tensor with the specified reduction applied.
        """
        # Ensure inputs are probabilities
        assert torch.all((inputs >= 0) & (inputs <= 1)), "Inputs must be probabilities in range [0, 1]."

        # Compute binary cross-entropy loss
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        # Compute p_t (probabilities aligned with targets)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Apply modulating factor
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction method
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: '{self.reduction}'. Choose from 'none', 'mean', or 'sum'.")


class DiceCrossEntropyLoss(nn.Module):
    """
    Dice + Cross-Entropy Loss for binary classification tasks with vector outputs.

    Args:
        dice_weight (float): Weight of the Dice loss in the combined loss. Default: 0.5.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'. Default: 'mean'.
        smooth (float): Smoothing factor to avoid division by zero in Dice loss. Default: 1.0.
    """
    def __init__(self, dice_weight: float = 0.5, reduction: str = "mean", smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice + Cross-Entropy Loss.

        Args:
            inputs (Tensor): Probabilities (sigmoid already applied) for each example (shape: [N]).
            targets (Tensor): Binary classification labels (0 or 1) with the same shape as inputs (shape: [N]).

        Returns:
            Tensor: Combined Dice + Cross-Entropy loss tensor with the specified reduction applied.
        """
        # Ensure inputs are probabilities
        assert torch.all((inputs >= 0) & (inputs <= 1)), "Inputs must be probabilities in range [0, 1]."

        # Compute binary cross-entropy loss
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        # Compute Dice loss
        intersection = (inputs * targets).sum()  # Batch-wise intersection
        union = inputs.sum() + targets.sum()  # Batch-wise union
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        # Combine losses
        loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss

        # Apply reduction method
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: '{self.reduction}'. Choose from 'none', 'mean', or 'sum'.")
        
