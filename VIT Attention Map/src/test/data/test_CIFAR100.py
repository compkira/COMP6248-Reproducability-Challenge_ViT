import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.main.data.CIFAR100 import trainset


if __name__ == "__main__":
    print(trainset.data.shape)
