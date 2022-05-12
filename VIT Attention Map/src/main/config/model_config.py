from src.main.config.config_base import ConfigBase


class ViTConfig(ConfigBase):
    model_name = "ViT"
    image_size = (3, 384, 384)  # TODO: put it in dataset's config
    patch_size = (16, 16)
    feature_dim = 768
    depth = 12
    num_heads = 12
    representation_size = None
    classifier = "token"
    num_classes = 1000
    qkv_bias = True
    hidden_dim_ratio = 4
    dropout_rate = 0.5


class VitAttentionScoreConfig(ConfigBase):
    model_name = "VitAttentionScore"
    image_size = (3, 384, 384)  # TODO: put it in dataset's config
    patch_size = (16, 16)
    feature_dim = 768
    depth = 12
    num_heads = 12
    representation_size = None
    classifier = "token"
    num_classes = 1000
    qkv_bias = True
    hidden_dim_ratio = 4
    dropout_rate = 0.1


class ResNetConfig(ConfigBase):
    model_name = "ResNet"
    feature_dim = 21
    num_block = 50
