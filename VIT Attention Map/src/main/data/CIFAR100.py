from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from src.main.config.config_factory import ConfigFactory


mc, _ = ConfigFactory()
image_size = (mc.image_size[1], mc.image_size[2])

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()  # convert to tensor
])

# load data
download = False if os.path.exists("src/main/data/cifar-100-python") else True
trainset = CIFAR100("src/main/data", train=True, download=download, transform=transform)
valset = CIFAR100("src/main/data", train=False, download=download, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
valloader = DataLoader(valset, batch_size=128, shuffle=True)
