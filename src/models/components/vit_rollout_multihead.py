from typing import Tuple, List
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class Vit(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        output_size: int = 1,
        return_nodes: str = "attn_drop",
        head_name: str = "head",
        img_size: int = 224,
    ) -> None:
        """Initialize the `Vit` module.

        Args:
            model_name (str): Name of the VIT.
            pretrained (bool, optional): Whether to use pretrained model. Defaults to True.
            output_size (int, optional): Number of classes. Defaults to 1.
            return_nodes (str, optional): Part of the model for Rollout calculation. Defaults to "attn_drop".
            head_name (str, optional): Classification head name. Defaults to "head".
            img_size (int, optional): Input image size. Defaults to 224.
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_size = output_size
        self.return_nodes = return_nodes
        self.head_name = head_name
        self.img_size = img_size
        self.model = self.__create_model()
        self.feature_extractor = self.__create_feature_extractor()

    def forward(self, input: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Perform a forward pass through the network.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Attention tensors wit classification output.
        """
        out = self.feature_extractor(input)
        attn_tensors = [v for k, v in out.items() if self.return_nodes in k]
        classification_out = out[self.head_name]
        return attn_tensors, classification_out

    def __create_model(self) -> nn.Module:
        """Creates model from timm library.

        Returns:
            nn.Module: Constructed model (Vit).
        """
        model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.output_size,
            img_size=self.img_size,
        )
        return model

    def __create_feature_extractor(self) -> nn.Module:
        """Creates feature extractor.

        Returns:
            nn.Module: Feature extractor.
        """
        for block in self.model.blocks:
            block.attn.fused_attn = False

        feature_layer_names = []
        for name, _ in self.model.named_modules():
            if "attn_drop" in name:
                feature_layer_names.append(name)

        # add classification output
        feature_layer_names.append(self.head_name)

        feature_extractor = create_feature_extractor(
            self.model, return_nodes=feature_layer_names
        )
        return feature_extractor


class AttentionRollout:
    def __init__(self, discard_ratio: float = 0.2, head_fusion: str = "mean") -> None:
        """Initialize the `AttentionRollout` module.

        Args:
            discard_ratio (float, optional): Percentage of attentions to drop. Defaults to 0.2.
            head_fusion (str, optional): Head fusion mode. Defaults to "mean".
        """
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def __call__(
        self, input_size: Tuple, attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform Attention Rollout and generate explainability map.

        Args:
            input_size (Tuple): Input size of model used.
            attentions (List[torch.Tensor]): List of attentions used for calculation.

        Raises:
            ValueError: If fusion type is not supproted.

        Returns:
            torch.Tensor: Explainability map.
        """
        # Use the same device as attentions
        device = attentions[0].device
        # Unzip shape of attentions
        batch_size, num_heads, height, width = attentions[0].shape
        # Create eye matrix the same shape as the attention matrix. It will be used as result.
        result = torch.eye(height, device=device).expand(batch_size, height, height)

        # Do not track gradients for computation
        with torch.no_grad():
            for attention in attentions:
                # Fuse attentions accros head dimension
                if self.head_fusion == "mean":
                    attention_heads_fused = attention.mean(dim=1)
                elif self.head_fusion == "max":
                    attention_heads_fused = attention.max(dim=1).values
                elif self.head_fusion == "min":
                    attention_heads_fused = attention.min(dim=1).values
                else:
                    raise ValueError("Attention head fusion type not supported")

                # Drop the lowest attentions, but don't drop the class token for each in the batch
                flat = attention_heads_fused.view(batch_size, -1)
                _, indices = flat.topk(
                    int(width * height * self.discard_ratio), dim=-1, largest=False
                )
                flat.scatter_(1, indices, 0)

                ''' The identity matrix serves as a baseline to incorporate self-attention
                into the aggregated attention. By adding it to the existing attention matrices,
                we ensure that each token retains some degree of self-focus, which is crucial
                for preserving the token's own information during the rollout.
                Division by 2 is required for balancing the contribution of the original
                attention and the self-attention.'''
                i_m = torch.eye(height, device=device).expand_as(attention_heads_fused)
                a = (attention_heads_fused + i_m) / 2
                a = a / a.sum(dim=-1, keepdim=True)

                # "Rolling" the attentions by multipying them sequentially
                result = torch.bmm(a, result)

        # Look at the total attention between the class token and the image patches for each in the batch
        map = result[:, 0, 1:]
        mask_width = int((height - 1) ** 0.5)
        map = map.view(batch_size, mask_width, mask_width)

        # Normalize
        map = map / map.view(batch_size, -1).max(dim=1).values.view(
            batch_size, 1, 1
        )
        map = map.unsqueeze(1)
        
        # Upscale explainability map to input image dimensions
        map = F.interpolate(
            map, size=input_size[2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        return map


class VitRolloutMultihead(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        output_size: int = 1,
        return_nodes: str = "attn_drop",
        head_name: str = "head",
        img_size: int = 224,
        discard_ratio: float = 0.2,
        head_fusion: str = "mean",
        multi_head: bool = False,
    ) -> None:
        """Initialize the `VitRolloutMultihead` module.

        Args:
            model_name (str, optional): Name of the VIT. Defaults to "vit_tiny_patch16_224.augreg_in21k_ft_in1k".
            pretrained (bool, optional): Whether to use pretrained model. Defaults to True.
            output_size (int, optional): Number of classes. Defaults to 1.
            return_nodes (str, optional): Part of the model for Rollout calculation. Defaults to "attn_drop".
            head_name (str, optional): Classification head name. Defaults to "head".
            img_size (float, optional): Input image size. Defaults to 224.
            discard_ratio (int, optional): Percentage of attentions to drop. Defaults to 0.2.
            head_fusion (str, optional): Head fusion mode. Defaults to "mean".
            multi_head (bool, optional): If True, the forward method returns both output and explainability output. Defaults to False.
        """
        super().__init__()
        self.model = Vit(
            model_name,
            pretrained=pretrained,
            output_size=output_size,
            return_nodes=return_nodes,
            head_name=head_name,
            img_size=img_size,
        )
        self.attention_rollout = AttentionRollout(
            discard_ratio=discard_ratio, head_fusion=head_fusion
        )
        self.multi_head = multi_head

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor containing the output, and optionally the Attention Rollout Map if
            multi_head is True.
        """
        attn_drop_tensors, classification_out = self.model(input)
        if self.multi_head:
            attn_mask = self.attention_rollout(input.size(), attn_drop_tensors)
            return classification_out, attn_mask
        else:
            return classification_out

def test_model():
    """Tests forward pass and prints out shapes
    """
    model = VitRolloutMultihead(multi_head=True)
    model.eval()
    dummy_input = torch.randn(10, 3, 224, 224)
    with torch.no_grad():
        out, cam = model(dummy_input)
    print("Output shape: ", out.shape)
    print("Explainability output shape: ", cam.shape)

if __name__ == "__main__":
    test_model()
