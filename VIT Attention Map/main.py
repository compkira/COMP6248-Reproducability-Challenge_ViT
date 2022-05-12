import torch
import timm

from src.main.models.vit import ViT
from src.main.train.train import Train
from src.main.config import ConfigFactory


mc, tc = None, None


def parse(**kwargs):
    mc, tc = ConfigFactory(**kwargs)
    mc.parse(**kwargs)
    tc.parse(**kwargs)

def train(**kwargs):
    parse(**kwargs)
    Train().train()

def finetune(**kwargs):
    parse(**kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire()
