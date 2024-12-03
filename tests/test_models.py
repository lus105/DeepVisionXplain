import hydra
import omegaconf
import torch
import pytest


@pytest.mark.parametrize('num_classes', [1, 10])
def test_simple_net(num_classes: int):
    cfg = {
        '_target_': 'src.models.components.simple_dense_net.SimpleDenseNet',
        'output_size': num_classes,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    batch_size = 2
    tensor = torch.randn((batch_size, 1, 28, 28))
    output = model.forward(tensor)

    assert output.size(dim=0) == batch_size
    assert output.size(dim=1) == num_classes


def test_cnn_cam_multihead():
    cfg = {
        '_target_': 'src.models.components.cnn_cam_multihead.CNNCAMMultihead',
        'backbone': 'torchvision.models/efficientnet_v2_s',
        'multi_head': 'True',
        'return_node': 'features.6.0.block.0',
        'weights': 'IMAGENET1K_V1',
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    batch_size = 2
    input_size = 224
    tensor = torch.randn((batch_size, 3, input_size, input_size))
    output = model.forward(tensor)

    assert output[0].size(dim=0) == batch_size
    assert output[1].size(dim=0) == batch_size
    assert output[1].size(dim=1) == input_size
    assert output[1].size(dim=2) == input_size


def test_vit_rollout_multihead():
    cfg = {
        '_target_': 'src.models.components.vit_rollout_multihead.VitRolloutMultihead',
        'backbone': 'timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        'multi_head': True,
        'pretrained': True,
        'output_size': 1,
        'return_nodes': 'attn_drop',
        'head_name': 'head',
        'img_size': 224,
        'discard_ratio': 0.2,
        'head_fusion': 'mean',
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    batch_size = 2
    input_size = 224
    tensor = torch.randn((batch_size, 3, input_size, input_size))
    output = model.forward(tensor)

    assert output[0].size(dim=0) == batch_size
    assert output[1].size(dim=0) == batch_size
    assert output[1].size(dim=1) == input_size
    assert output[1].size(dim=2) == input_size
