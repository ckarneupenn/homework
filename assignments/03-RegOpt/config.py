from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    """
    Create a new config.

    Note to students: You can change the arguments to this constructor,
    if you need to add new parameters.

    """

    batch_size = 64
    num_epochs = 5
    initial_learning_rate = 0.002
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "drop_point": 200,
        "initial_learning_rate": initial_learning_rate,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )