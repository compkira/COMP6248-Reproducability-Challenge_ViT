import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Embedding(nn.Module):
    def __init__(self, image_size, patch_size, feature_dim, dropout_rate=0.0, pool='cls'):
        super().__init__()

        channels, image_height, image_width = image_size
        patch_height, patch_width = patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pool = pool

        self.patch_enbedding = nn.Sequential(
            nn.Conv2d(channels, feature_dim, patch_size, patch_size),
            Rearrange('B C P1 P2 -> B (P1 P2) C'),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.pos_embedding = nn.Parameter(torch.normal(0, 0.02, size=(1, num_patches + 1, feature_dim)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_enbedding(x)
        B, N, _ = x.shape

        cls_token = repeat(self.cls_token, '1 N D -> B N D', B=B)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)

        return x


class Attention(nn.Module):
    def __init__(self, feature_dim, num_heads, qkv_bias=True, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_heads = feature_dim // num_heads
        self.scale = (self.dim_heads ** -0.5)

        self.qkv = nn.Linear(feature_dim, 3 * feature_dim, bias=qkv_bias)
        self.attn_weight = nn.Softmax(dim=-1)
        self.attn_weight_dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'B N (h D) -> B h N D', h = self.num_heads), qkv)

        attn_score = q @ k.transpose(-1, -2) * self.scale
        attn_weight = self.attn_weight(attn_score) # B h N N
        attn_weight = self.attn_weight_dropout(attn_weight) # B h N N
        weighted_feature = attn_weight @ v # B h N D
        weighted_feature = rearrange(weighted_feature, 'B h N D -> B N (h D)')
        return self.proj(weighted_feature)


class AttentionScore(nn.Module):
    def __init__(self, feature_dim, num_heads, qkv_bias=True, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_heads = feature_dim // num_heads
        self.scale = (self.dim_heads ** -0.5)

        self.qkv = nn.Linear(feature_dim, 3 * feature_dim, bias=qkv_bias)
        self.attn_weight = nn.Softmax(dim=-1)
        self.attn_weight_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'B N (h D) -> B h N D', h = self.num_heads), qkv)

        attn_score = q @ k.transpose(-1, -2) * self.scale
        attn_weight = self.attn_weight(attn_score) # B h N N
        return attn_weight


class MLPBlock(nn.Sequential):
    def __init__(self, feature_dim, hidden_dim, dropout_rate=0.0):
        super().__init__(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout_rate)
        )


class ResidualBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)


class Transformer(nn.Sequential):
    def __init__(self, depth, feature_dim, num_heads, qkv_bias=True, hidden_dim_ratio=4, dropout_rate=0.0):
        

        net = nn.Sequential()
        for i in range(depth):
            net.add_module(f"Transformer_{i}_Attention",
                ResidualBlock(nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    Attention(feature_dim, num_heads, qkv_bias, dropout_rate),
                    # nn.Dropout(dropout_rate)
                ))
            )

            net.add_module(f"Transformer_{i}_MLPBlock",
                ResidualBlock(nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    MLPBlock(feature_dim, int(feature_dim * hidden_dim_ratio), dropout_rate)
                ))
            )
        # self.net = ResidualBlock(nn.Linear(feature_dim, hidden_dim))
        

        net.add_module("Transformer_output_LayerNorm", nn.LayerNorm(feature_dim))
        super().__init__(net)


class MLPHead(nn.Sequential):
    def __init__(self, representation_size, feature_dim, num_classes):
        if representation_size is not None:
            proj = nn.Sequential(
                nn.Linear(feature_dim, representation_size),
                nn.Tanh(),
                nn.Linear(representation_size, num_classes)
            )
        else:
            proj = nn.Linear(feature_dim, num_classes)

        super().__init__(proj)


class VitAttentionScore(nn.Module):
    def __init__(self, image_size, patch_size, feature_dim, depth, num_heads, representation_size, classifier, num_classes, qkv_bias=True, hidden_dim_ratio=4.0, dropout_rate=0.0):
        super().__init__()

        self.embedding = Embedding(image_size, patch_size, feature_dim, dropout_rate)
        self.transformer = Transformer(depth-1, feature_dim, num_heads, qkv_bias, hidden_dim_ratio, dropout_rate)
        self.attentionScore = AttentionScore(feature_dim, num_heads, qkv_bias, dropout_rate)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.attentionScore(x)


if __name__ == '__main__':
    import numpy as np
    import timm


    def get_n_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def assert_tensors_equal(t1, t2):
        a1, a2 = t1.detach().numpy(), t2.detach().numpy()

        np.testing.assert_allclose(a1, a2)

    model_name = "vit_base_patch16_384"
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    print(type(model_official))

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

    model_custom = VitAttentionScore(**custom_config)
    model_custom.eval()

    for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
        assert p_o.numel() == p_c.numel()
        print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data
        assert_tensors_equal(p_c.data, p_o.data)

    print("---------------------------------------------------------------------------")

    inp = torch.rand(1, 3, 384, 384)
    res_c = model_custom(inp)
    res_o = model_official(inp)

    assert get_n_params(model_custom) == get_n_params(model_official)
    print("---------------------------------------------------------------------------")
    assert_tensors_equal(res_c, res_o)
