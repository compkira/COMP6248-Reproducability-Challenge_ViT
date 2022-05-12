from src.main.config.model_config import ViTConfig, ResNetConfig, VitAttentionScoreConfig
from src.main.config.train_config import TrainConfig


class ConfigFactory:
    def __new__(cls, **kwargs):
        if not kwargs:
            model_name = "ViT"
        elif "model_name" not in kwargs.keys():
            model_name = "ViT"
        else:
            model_name = kwargs["model_name"]

        if model_name == "ViT":
            mc = ViTConfig()
        elif model_name == "ResNet":
            mc = ResNetConfig()
        elif model_name == "VitAttentionScore":
            mc = VitAttentionScoreConfig()
        else:
            raise ValueError(f'Invalid moddel={model_name}')

        tc = TrainConfig()

        return mc, tc
