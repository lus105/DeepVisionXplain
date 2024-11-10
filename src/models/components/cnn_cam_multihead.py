from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

from nn_utils import create_model
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model: nn.Module, return_node: str, out_name: str = "layerout") -> None:
        """Initialize a FeatureExtractor module.

        Args:
            pretrained_model (nn.Module): model with pretrained weights.
            return_node (str): node of the model that is considered as output.
            out_name (str, optional): output layer name. Defaults to "layerout".

        Raises:
            ValueError: if no return nodes are specified
        """
        super().__init__()
        try:
            if not return_node:
                log.error("No return_node provided to FeatureExtractor")
                raise ValueError("return_nodes must contain at least one node.")

            self.out_name = out_name
            return_nodes = {return_node: out_name}

            self.model = create_feature_extractor(
                pretrained_model, return_nodes=return_nodes
            )
            self.n_features = self._calculate_n_features()
        except Exception as e:
            log.exception("Failed to initialize FeatureExtractor")
            raise e

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        features = self.model(input)
        return features[self.out_name]

    def _calculate_n_features(self, input_shape: tuple=(1, 3, 224, 224)) -> int:
        """Calculates the number of channel features in out layer.

        Args:
            input_shape (tuple, optional): Dummy input shape. Defaults to (1, 3, 224, 224).

        Returns:
            int: Number of channel features in out layer.
        """
        with torch.no_grad():
            return self.forward(torch.zeros(input_shape)).shape[1]


class BinaryClassificationHead(nn.Module):
    """Neural network head for binary classification.

    It consists of a global average pooling layer followed by a fully connected
    layer.
    """
    def __init__(self, last_layer_features: int, num_classes: int = 1):
        """Initialize the `BinaryClassificationHead` module.

        Args:
            last_layer_features (int): The number of features in the last layer of the base model.
            num_classes (int, optional): Number of classes. Defaults to 1.
        """
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_layer_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor of predictions of shape (batch_size, 1).
        """
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ClassActivationMapGenerator(nn.Module):
    """Generates Class Activation Maps (CAM) based on given features and weights."""

    def __init__(self, fc: nn.Linear):
        """Initialize the `ClassActivationMapGenerator` module.

        Args:
            fc (nn.Linear): A Linear layer (typically from a classifier) whose weights will be used
            to generate CAM.
        """
        super().__init__()
        self.fc = fc

    def forward(self, input_size: tuple, features: torch.Tensor) -> torch.Tensor:
        """Compute the Class Activation Maps using features and weights from the fc layer.

        Args:
            input_size (Tuple): Entire model input size.
            features (torch.Tensor): Feature maps from which CAMs will be generated.

        Returns:
            torch.Tensor: A tensor containing the Class Activation Maps of shape (B, H, W).
        """
        weights = self.fc.weight.detach().unsqueeze(-1).unsqueeze(-1)
        cam = torch.einsum("ijkl,ijkl->ikl", features, weights).unsqueeze(
            1
        )  # (B, 1, H, W)

        cam = F.interpolate(
            cam, size=input_size[2:], mode="bilinear", align_corners=False
        ).squeeze(1)  # (B, H, W)

        return cam


class CNNCAMMultihead(nn.Module):
    """Extends the CNN model to produce binary classification and optionally generate
    Class Activation Maps (CAM)."""

    def __init__(
        self,
        backbone: str,
        multi_head: bool = False,
        return_node: str = "features.16",
        weights: str = "IMAGENET1K_V1",
    ):
        """Initialize the `CNNCAMMultihead` module.

        Args:
            backbone (str, optional): Model backbone used. Defaults to "mobilenet_v3_large".
            multi_head (bool, optional): If True, the forward method returns both output and CAM.
            Otherwise, it returns only the output. Defaults to False.
            return_node (str, optional): Return node of the feature extractor. Defaults to "features.16".
            weights (str, optional): Pre-trained weights to be used with the model. Defaults to "IMAGENET1K_V1".

        Raises:
            ValueError: If no valid backbone is found
        """
        super().__init__()
        self.multi_head = multi_head

        pretrained_model = create_model(backbone, weights=weights)
        
        self.feature_extractor = FeatureExtractor(
            pretrained_model, return_node=return_node
        )
        self.output_layer = BinaryClassificationHead(
            last_layer_features=self.feature_extractor.n_features
        )
        self.cam_generator = ClassActivationMapGenerator(self.output_layer.fc)

    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Perform a forward pass through the network.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: A tensor containing the output,
              and optionally the Class Activation Map if multi_head is True.
        """
        features = self.feature_extractor(input)
        output = self.output_layer(features)
        output = torch.sigmoid(output)

        if self.multi_head:
            cam = self.cam_generator(input.size(), features)
            return output, cam
        else:
            return output