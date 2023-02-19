from typing import List
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Create a new scheduler.
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        drop_point=0.1,
        initial_learning_rate=0.1,
    ):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        The scheduler changes the learning rate every 200 batches using the formula:
        new lr = old lr * exp(self.k * self.last_epoch) - 0.04
        where k is a hyperparameter with value 0.00001.
        """
        # ... Your Code Here ...
        self.drop_point = drop_point

        self.initial_learning_rate = initial_learning_rate

        self.k = 0.00001

        self.func = lambda x: x * ((np.exp(self.k * self.last_epoch)) - 0.04)

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        runs step() on the scheduler

        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        if self.last_epoch == 0:
            return [self.initial_learning_rate]

        if self.last_epoch % self.drop_point == 0:
            self.initial_learning_rate = self.func(self.initial_learning_rate)
            return [self.initial_learning_rate]

        return [self.initial_learning_rate]
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
