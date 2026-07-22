import numpy as np

from src.model.layer import DenseLayer


class Network:
    def __init__(self, layers: tuple[DenseLayer, ...]):
        self.layers = layers

    def _initialize(self, n_in: int):
        for layer in self.layers:
            layer.initialize(n_in)
            n_in = layer.n_neurons

    def forward(self, X_train: np.ndarray) -> np.ndarray:
        prev_A = X_train
        print(prev_A.shape)
        for layer in self.layers:
            layer.forward(prev_A)
            prev_A = layer.A
            print(prev_A.shape)
        
        return prev_A
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        dA = y_pred - y_true
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_valid: np.ndarray,
            y_valid: np.ndarray,
            epochs: int,
            learning_rate: float
        ):
        self._initialize(X_train.shape[0])

        for i in range(epochs):
            y_pred = self.forward(X_train)
            self.backward(y_train, y_pred, learning_rate)

    def predict(self, X) -> np.ndarray:
        return self.forward(X)