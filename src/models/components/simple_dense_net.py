import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        Args:
            input_size (int, optional): The number of input features.. Defaults to 784.
            lin1_size (int, optional): The number of output features of the first linear layer. Defaults to 256.
            lin2_size (int, optional): The number of output features of the second linear layer. Defaults to 256.
            lin3_size (int, optional): The number of output features of the third linear layer. Defaults to 256.
            output_size (int, optional): The number of output features of the final linear layer. Defaults to 10.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
