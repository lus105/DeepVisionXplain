import hydra
import omegaconf
import torch
import pytest


@pytest.mark.parametrize("num_classes", [1, 10])
def test_simple_net(num_classes: int):
    cfg = {
        "_target_": "src.models.components.simple_dense_net.SimpleDenseNet",
        "output_size": num_classes,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    batch_size = 2
    tensor = torch.randn((batch_size, 1, 28, 28))
    output = model.forward(tensor)

    assert output.size(dim=0) == batch_size
    assert output.size(dim=1) == num_classes
