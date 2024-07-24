import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, efficientnet_v2_s
from torchvision.models.feature_extraction import create_feature_extractor


class FeatureExtractor(nn.Module):
    """Initialize Feature extractor."""

    def __init__(self, pretrained_model, return_nodes: dict) -> None:
        """Initialize a `FeatureExtractor` module.

        :param pretrained_model: model with pretrained weights.
        :param return_nodes: node of the model that is considered as output.
        """
        super().__init__()

        if not return_nodes:
            raise ValueError("return_nodes must contain at least one node.")

        nodes = list(return_nodes.values())
        self.out_name = nodes[0]

        self.model = create_feature_extractor(
            pretrained_model, return_nodes=return_nodes
        )
        self.n_features = self._calculate_n_features()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param input: The input tensor.
        :return: A tensor of predictions.
        """
        features = self.model(input)
        return features[self.out_name]

    def _calculate_n_features(self):
        dummy_input = torch.randn(1, 3, 100, 100)
        with torch.no_grad():
            output = self.forward(dummy_input)
        return output.shape[1]


class BinaryClassificationHead(nn.Module):
    """Neural network head for binary classification.

    It consists of a global average pooling layer followed by a fully connected
    layer.
    """

    def __init__(self, last_layer_features: int):
        """Initialize the `BinaryClassificationHead` module.

        :param last_layer_features: The number of features in the last layer of the base network.
        """
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_layer_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        :param x: The input tensor of shape (batch_size, channels, height, width).
        :return: A tensor of predictions of shape (batch_size, 1).
        """
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ClassActivationMapGenerator(nn.Module):
    """Generates Class Activation Maps (CAM) based on given features and weights."""

    def __init__(self, fc: nn.Linear):
        """Initialize the `ClassActivationMapGenerator` module.

        :param fc: A Linear layer (typically from a classifier) whose weights will be used
            to generate CAM.
        """
        super().__init__()
        self.fc = fc

    def forward(self, input: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Compute the Class Activation Maps using features and weights from the fc layer.

        :param input: Original input tensor to the entire model. Used for its size to resize the
            CAMs.
        :param features: Feature maps from which CAMs will be generated.
        :return: A tensor containing the Class Activation Maps of shape (B, H, W).
        """
        weights = self.fc.weight.data.unsqueeze(-1).unsqueeze(-1)
        cam = torch.einsum("ijkl,ijkl->ikl", features, weights).unsqueeze(
            1
        )  # (B, 1, H, W)

        cam = F.interpolate(
            cam, size=input.size()[2:], mode="bilinear", align_corners=False
        ).squeeze(1)  # (B, H, W)

        return cam


class CNNCAMMultihead(nn.Module):
    """Extends the CNN model to produce binary classification and optionally generate
    Class Activation Maps (CAM)."""

    def __init__(
        self,
        backbone: str = "mobilenet_v3_large",
        multi_head: bool = False,
        return_nodes: dict = {"features.5.1.block.2": "layerout"},
        weights: str = "IMAGENET1K_V1",
    ):
        """Initialize the `CNNCAMMultihead` module.

        :param multi_head: If True, the forward method returns both output and CAM.
            Otherwise, it returns only the output.
        :param return_nodes: Dictionary for the return nodes used in feature extraction.
        :param last_layer_features: The number of features in the last layer of the base network.
        :param weights: Pre-trained weights to be used with the efficientnet_b3 model.
        """
        super().__init__()
        if backbone == "mobilenet_v3_large":
            pretrained_model = mobilenet_v3_large(wights=weights)
        elif backbone == "efficientnet_v2_s":
            pretrained_model = efficientnet_v2_s(weights=weights)
        self.feature_extractor = FeatureExtractor(
            pretrained_model, return_nodes=return_nodes
        )
        self.output_layer = BinaryClassificationHead(
            last_layer_features=self.feature_extractor.n_features
        )
        self.cam_generator = ClassActivationMapGenerator(self.output_layer.fc)
        self.multi_head = multi_head

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        :param input: The input tensor of shape (batch_size, channels, height, width).
        :return: A tensor containing the output, and optionally the Class Activation Map if
            multi_head is True.
        """
        features = self.feature_extractor(input)
        output = self.output_layer(features)

        if not self.multi_head:
            return output
        else:
            cam = self.cam_generator(input, features)
            return output, cam


if __name__ == "__main__":
    _ = CNNCAMMultihead()
