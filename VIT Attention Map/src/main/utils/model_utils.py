from json import load
import os
import torch
from torch import nn

from src.main.config.config_factory import ConfigFactory


mc, tc = ConfigFactory()

# def load_from(path):
#     print(f"loaded from {path}")

# def save_to(path):
#     print(f"saved to {path}")

def initialize_model(model):
    @torch.no_grad()
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    model.apply(weights_init)

def load_model(model):
    if os.path.exists(f"src/main/checkpoints/models/{mc.model_name}.pkl"):
        model.load_state_dict(torch.load(f"src/main/checkpoints/models/{mc.model_name}.pkl"))
        # load_from(f"src/main/checkpoints/models/{model_config.model_name}.pkl")
    # else:
    #     @torch.no_grad()
    #     def weights_init(m):
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    #     model.apply(weights_init)

def load_optimizer(optimizer):
    if os.path.exists(f"src/main/checkpoints/optimizers/{mc.model_name}.pkl"):
        optimizer.load_state_dict(torch.load(f"src/main/checkpoints/optimizers/{mc.model_name}.pkl"))
        # load_from(f"src/main/checkpoints/optimizers/{model_config.model_name}.pkl")

def load_scheduler(scheduler):
    if os.path.exists(f"src/main/checkpoints/scheduler/{mc.model_name}.pkl"):
        scheduler.load_state_dict(torch.load(f"src/main/checkpoints/scheduler/{mc.model_name}.pkl"))
        # load_from(f"src/main/checkpoints/scheduler/{model_config.model_name}.pkl")

def save_model(model):
    torch.save(model.state_dict(), f"src/main/checkpoints/models/{mc.model_name}.pkl")
    # save_to(f"src/main/checkpoints/models/{model_config.model_name}.pkl")

def save_optimizer(optimizer):
    torch.save(optimizer.state_dict(), f"src/main/checkpoints/optimizers/{mc.model_name}.pkl")
    # save_to(f"src/main/checkpoints/optimizers/{model_config.model_name}.pkl")

def save_scheduler(scheduler):
    torch.save(scheduler.state_dict(), f"src/main/checkpoints/schedulers/{mc.model_name}.pkl")
    # save_to(f"src/main/checkpoints/schedulers/{model_config.model_name}.pkl")