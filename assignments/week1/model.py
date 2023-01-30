import numpy as np
from numpy import random
import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        pass

    def fit(self, X, y):
        # raise NotImplementedError()
        if np.linalg.det(X.T@X) != 0:
          self.w = (np.linalg.inv(X.T@X))@(X.T@y)
        else:
          print('Closed form solution not possible here as determinant of X.T@X is not zero')

    def predict(self, X):
        # raise NotImplementedError()
        self.b = np.ones((X.shape[0], ))
        preds = (X@self.w.T) + self.b
        return preds


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # raise NotImplementedError()
        
        X = torch.tensor(X)
        y = torch.tensor(y)

        model = nn.Sequential(nn.Linear(in_features=8, out_features=1).double())

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        losses = []

        for i in range(0, epochs):

          preds = model(X)
          preds = preds.squeeze(1)
          loss = criterion(preds, y)

          if i%100 == 0:
            print(f'Loss in {i}th epoch is {loss.item()}')

          losses.append(loss.item())
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

        print('Final loss is ', losses[-1])

        self.w = model[0].weight.data.squeeze(0)
        print('The final trained weights are: ', self.w)

        fig, ax = plt.subplots()
        ax = plt.plot(range(epochs), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.show
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # raise NotImplementedError()

        X = torch.tensor(X)
        preds = X@self.w.T
        return preds.numpy()
