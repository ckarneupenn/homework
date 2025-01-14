from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    """
    CONFIG Class containing all required hyperparameters
    """

    batch_size = 128
    num_epochs = 1

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.0015)

    transforms = Compose([ToTensor()])
