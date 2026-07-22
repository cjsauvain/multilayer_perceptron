import numpy as np
from src.model.activation import sigmoid, sigmoid_deriv, relu, relu_deriv
from src.model.initializer import he_normal_initializer, glorot_normal_initializer


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu": (relu, relu_deriv)
}
INITIALIZERS = {
    "HeNormal": he_normal_initializer,
    "GlorotNormal": glorot_normal_initializer
}


class DenseLayer:
    def __init__(self, n_neurons: int, activation: str = "relu", weight_initializer: str = "HeNormal"):
        self.n_neurons = n_neurons
        self.activation_fn, self.activation_deriv = ACTIVATIONS[activation]
        self.A = None
        self.prev_A = None
        self.Z = None
        self.weights_initializer = weight_initializer
        self.W = None
        self.b = None
    
    def __repr__(self) -> str:
        return (
            f"n_neurons= {self.n_neurons}, "
            f"activation_fn= {self.activation_fn.__name__}, "
            f"activation_deriv= {self.activation_deriv.__name__}, "
            f"weights_initializer= {self.weights_initializer}"
        )
    
    def initialize(self, n_in: int):
        initializer_fn = INITIALIZERS[self.weights_initializer]
        self.W = initializer_fn(n_in, self.n_neurons)
        self.b = np.zeros((self.n_neurons, 1))

    def forward(self, prev_A: np.ndarray):
        self.A, self.Z = self.activation_fn(prev_A, self.W, self.b)
        self.prev_A = prev_A

    def backward(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
        m = self.prev_A.shape[0]
        dZ = dA * self.activation_deriv(self.A)
        dA_prev = self.W.T.dot(dZ)
        
        dW = (1/m) * dZ.dot(self.prev_A.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev