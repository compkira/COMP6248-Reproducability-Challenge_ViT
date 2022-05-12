import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.main.config.model_config import ModelConfig
import inspect


if __name__ == "__main__":
    print(ModelConfig().to_dict())
