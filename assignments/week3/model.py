import torch
from typing import Callable


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """

        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(1, hidden_count + 1):
            if i == 1:
                self.layers += [
                    torch.nn.Linear(in_features=input_size, out_features=hidden_size)
                ]
                self.layers += [activation()]
            else:
                self.layers += [
                    torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
                ]
                self.layers += [activation()]

        self.out = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)

        # Weight Initialization

        for i in range(len(list(self.layers))):
            if isinstance(list(self.layers)[i], torch.nn.Linear):
                initializer(list(self.layers)[i].weight.data)

        initializer(self.out.weight.data)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x
