import torch


class Model(torch.nn.Module):
    """
    This is the Model class that has the CNN
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Create all CNN layers required
        """
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels, out_channels=32, kernel_size=4, stride=2
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc1 = torch.nn.Linear(in_features=800, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass x through all layers
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(1, -1)
        x = self.fc1(x)

        return x
