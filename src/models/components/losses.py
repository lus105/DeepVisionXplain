import torch
import torch.nn.functional as F


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Focal loss for dense detection tasks, such as RetinaNet.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                Probabilities (sigmoid already applied) for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Binary
                classification labels (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: 2.
        reduction (str): ``'none'`` | ``'mean'`` | ``'sum'``.
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.

    Returns:
        Tensor: Loss tensor with the specified reduction applied.
    """
    # Ensure inputs are probabilities (sigmoid should already be applied)
    assert torch.all((inputs >= 0) & (inputs <= 1)), "Inputs must be probabilities in range [0, 1]."

    # Compute binary cross-entropy loss
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

    # Compute p_t (probabilities aligned with targets)
    p_t = inputs * targets + (1 - inputs) * (1 - targets)

    # Apply modulating factor
    focal_term = (1 - p_t) ** gamma
    loss = focal_term * ce_loss

    # Apply alpha weighting if specified
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction method
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: '{reduction}'. Choose from 'none', 'mean', or 'sum'.")

if __name__ == '__main__':
    inputs = torch.tensor([0.1, 0.9, 0.8, 0.3], requires_grad=True)  # Probabilities (sigmoid applied)
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.float32)  # Binary labels
    loss = focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="mean")
    print("Focal Loss:", loss.item())