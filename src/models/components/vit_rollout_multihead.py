import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class Vit(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained: bool = True,
        weight_path: str = None,
        output_size: int = 1,
        return_nodes: str = "attn_drop",
        head_name: str = "head",
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.weight_path = weight_path
        self.output_size = output_size
        self.return_nodes = return_nodes
        self.head_name = head_name
        self.img_size = img_size
        self.model = self.__create_model()
        self.feature_extractor = self.__create_feature_extractor()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(input)
        attn_drop_tensors = [v for k, v in out.items() if self.return_nodes in k]
        classification_out = out[self.head_name]
        return attn_drop_tensors, classification_out

    def __create_model(self) -> nn.Module:
        model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.output_size,
            img_size=self.img_size,
        )
        if self.weight_path is not None:
            model.load_state_dict(torch.load(self.weight_path))
        return model

    def __create_feature_extractor(self) -> nn.Module:
        for block in self.model.blocks:
            block.attn.fused_attn = False

        feature_layer_names = []
        for name, i in self.model.named_modules():
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
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def __call__(
        self, input: torch.Tensor, attentions: list[torch.Tensor]
    ) -> torch.Tensor:
        device = attentions[0].device
        batch_size, num_heads, height, width = attentions[0].shape
        # Create eye matrix the same shape as the attention matrix for each item in the batch
        result = torch.eye(height, device=device).expand(batch_size, height, height)

        with torch.no_grad():
            for attention in attentions:
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

                I = torch.eye(height, device=device).expand_as(attention_heads_fused)
                a = (attention_heads_fused + I) / 2
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.bmm(a, result)

        # Look at the total attention between the class token and the image patches for each in the batch
        masks = result[:, 0, 1:]
        mask_width = int((height - 1) ** 0.5)
        masks = masks.view(batch_size, mask_width, mask_width)
        masks = masks / masks.view(batch_size, -1).max(dim=1).values.view(
            batch_size, 1, 1
        )

        masks = masks.unsqueeze(1)

        masks = F.interpolate(
            masks, size=input.size()[2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        return masks


class VitRolloutMultihead(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        weight_path: str = None,
        output_size: int = 1,
        return_nodes: str = "attn_drop",
        head_name: str = "head",
        img_size: int = 224,
        discard_ratio: int = 0.2,
        head_fusion: str = "mean",
        visualize: bool = False,
    ) -> None:
        super().__init__()
        self.model = Vit(
            model_name,
            pretrained=pretrained,
            weight_path=weight_path,
            output_size=output_size,
            return_nodes=return_nodes,
            head_name=head_name,
            img_size=img_size,
        )
        self.attention_rollout = AttentionRollout(
            discard_ratio=discard_ratio, head_fusion=head_fusion
        )
        self.visualize = visualize

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn_drop_tensors, classification_out = self.model(input)
        if self.visualize:
            attn_mask = self.attention_rollout(input, attn_drop_tensors)
            return classification_out, attn_mask
        else:
            return classification_out


if __name__ == "__main__":
    _ = VitRolloutMultihead()
