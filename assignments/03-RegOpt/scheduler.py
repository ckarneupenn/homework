from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, drop=0.77, drop_point=0.1, initial_learning_rate=0.1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.drop = drop
        self.drop_point = drop_point
        self.initial_learning_rate = initial_learning_rate
        
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        if self.last_epoch == 0:
          return [self.initial_learning_rate]

        if self.last_epoch % 50 == 0:
          self.initial_learning_rate = self.initial_learning_rate * (self.drop)
          return [self.initial_learning_rate]

        return [self.initial_learning_rate]
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
