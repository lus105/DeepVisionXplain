import torch
import torch.nn as nn
import timm

class Vit(nn.Module):
    def __init__(
            self,
            model_name,
            pretrained: bool = True,
            weight_path: str = None,
            output_size: int = 1,
            return_nodes: str = 'attn_drop'
            ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.weight_path = weight_path
        self.output_size = output_size
        self.return_nodes = return_nodes
        self.model = self.__create_model()
        self.__customize_classifier()
        self.feature_outputs = []  # For storing features from hooks
        self.__attach_hooks()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.model(input)
        features = self.feature_outputs
        return features, out 
    
    def __create_model(self) -> nn.Module:
        model = timm.create_model(self.model_name, pretrained=self.pretrained)
        if self.weight_path is not None:
            model.load_state_dict(torch.load(self.weight_path))
        return model
    
    def __customize_classifier(self) -> None:
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, self.output_size)
    
    def __attach_hooks(self) -> nn.Module:
        for block in self.model.blocks:
            block.attn.fused_attn = False
        
        def hook_fn(module, input, output):
            self.feature_outputs.append(output)

        # Traverse the model and attach hooks to the layers of interest
        for name, layer in self.model.named_modules():
            if self.return_nodes in name:
                layer.register_forward_hook(hook_fn)


class AttentionRollout():
    def __init__(
            self,
            discard_ratio: int = 0.2,
            head_fusion: str = "mean"
            ) -> None:
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def __call__(self, attentions: list[torch.Tensor]) -> torch.Tensor:
        device = attentions[0].device
        # create eye matrix the same sahpe as the attention matrix
        result = torch.eye(attentions[0].size(-1), device=device)
        with torch.no_grad():
            for attention in attentions:
                if self.head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif self.head_fusion == "max":
                   attention_heads_fused = attention.max(axis=1)[0]
                elif self.head_fusion == "min":
                   attention_heads_fused = attention.min(axis=1)[0]
                else:
                   raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1)).to(device)
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
        
        # Look at the total attention between the class token and the image patches
        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width)
        mask = mask / mask.max()
        return mask


class VitRolloutMultihead(nn.Module):
    def __init__(
            self,
            model_name: str = "vit_tiny_patch16_224.augreg_in21k",
            pretrained: bool = True,
            weight_path: str = None,
            output_size: int = 1,
            return_nodes: str = 'attn_drop',
            discard_ratio: int = 0.2,
            head_fusion: str = "mean",
            visualize: bool = False,
            ) -> None:
        super().__init__()
        self.model = Vit(model_name,
                         pretrained=pretrained,
                         weight_path=weight_path,
                         output_size=output_size,
                         return_nodes=return_nodes)
        self.attention_rollout = AttentionRollout(discard_ratio=discard_ratio,
                                                  head_fusion=head_fusion)
        self.visualize = visualize

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features, out =  self.model(input)
        out =  self.model(input)
        if self.visualize:
            attn_mask = self.attention_rollout(features)
            return out, attn_mask
        else:
            return out

if __name__ == "__main__":
    _ = VitRolloutMultihead()
    # model = VitRolloutMultihead(visualize=True).to("cuda")
    # dummy_input = torch.randn((1, 3, 224, 224)).to("cuda")
    # x, out = model(dummy_input)
    # print(x)
    # print(out)
