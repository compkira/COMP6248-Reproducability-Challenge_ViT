import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# print(sys.path)

import numpy as np
import timm

from src.main.models.vit import *


device = "cpu"

def test_ViT():
    def get_n_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def assert_tensors_equal(t1, t2):
        a1, a2 = t1.detach().numpy(), t2.detach().numpy()

        np.testing.assert_allclose(a1, a2)

    model_name = "vit_base_patch16_384"
    model_official = timm.create_model(model_name, pretrained=True).to(device)
    model_official.eval()
    print(model_official)

    custom_config = {
        # image_size, patch_size, feature_dim, depth, num_heads, representation_size, classifier, num_classes
        "image_size": (3, 384, 384),
        "patch_size": (16, 16),
        "feature_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "representation_size": None,
        "classifier": "token",
        "num_classes": 1000
    }

    model_custom = ViT(**custom_config).to(device)
    model_custom.eval()

    for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
        assert p_o.numel() == p_c.numel()
        print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data
        # assert_tensors_equal(p_c.data, p_o.data)

    print("---------------------------------------------------------------------------")

    inp = torch.rand(1, 3, 384, 384).to(device)
    res_c = model_custom(inp)
    res_o = model_official(inp)

    assert get_n_params(model_custom) == get_n_params(model_official)
    print("---------------------------------------------------------------------------")
    assert_tensors_equal(res_c, res_o)


if __name__ == "__main__":
    test_ViT()


