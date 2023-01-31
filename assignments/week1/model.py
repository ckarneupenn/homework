import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        pass

    def fit(self, X, y):
        # raise NotImplementedError()
        if np.linalg.det(X.T @ X) != 0:
            self.w = (np.linalg.inv(X.T @ X)) @ (X.T @ y)
        else:
            print(
                "Closed form solution not possible here as determinant of X.T@X is not zero"
            )

    def predict(self, X):
        # raise NotImplementedError()
        self.b = np.zeros((X.shape[0],))
        preds = (X @ self.w.T) + self.b
        return preds


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
        # raise NotImplementedError()

        self.w = np.random.randn(X.shape[1], 1)  # initial weights
        y = y.reshape(y.shape[0], 1)
        losses = []

        for i in range(0, epochs):

            preds = X @ self.w
            loss = np.sum((preds - y) ** 2)
            losses.append(loss)
            if i % 100 == 0:
                print(f"Loss in epoch {i} is {loss}")

            gradients = 2 * (X.T.dot(preds - y)) / X.shape[0]
            self.w = self.w - lr * gradients

    def predict(self, X: np.ndarray):
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # raise NotImplementedError()

        preds = X @ self.w
        return preds
