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
        max_lr=0.005,
        base=0.001,
    ):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        The scheduler changes the learning rate every 200 batches using the cyclical learning
        rate scheduler but I modified it:
        """
        # ... Your Code Here ...
        self.drop_point = drop_point
        self.max_lr = max_lr
        self.base = base
        self.dummy = 0

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        runs step() on the scheduler

        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        if self.last_epoch == 0:
            return [self.base_lrs[0]]

        for i in range(0, 10000):

            if self.last_epoch == 2400:
                self.dummy = 2

            if self.dummy == 2:
                self.base = self.base - ((self.max_lr - self.base_lrs[0]) / 10000)
                return [self.base]

            if self.dummy == 0:

                self.base = self.base + (
                    (self.max_lr - self.base_lrs[0]) / self.drop_point
                )
                if int(self.last_epoch) % self.drop_point == 0:
                    self.dummy = 1
                return [self.base]

            if self.dummy == 1:

                self.base = self.base - (
                    (self.max_lr - self.base_lrs[0]) / self.drop_point
                )
                if int(self.last_epoch) % self.drop_point == 0:
                    self.dummy = 0
                return [self.base]
            
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
