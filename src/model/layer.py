import numpy as np

from src.model.activation import (
    relu,
    relu_deriv,
    sigmoid,
    sigmoid_deriv,
    softmax,
)
from src.model.initializer import (
    glorot_normal_initializer,
    he_normal_initializer,
)

ACTIVATIONS = {
    "softmax": (softmax, None),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu": (relu, relu_deriv),
}
INITIALIZERS = {
    "HeNormal": he_normal_initializer,
    "GlorotNormal": glorot_normal_initializer,
}


class DenseLayer:
    """A fully connected layer with its own weights, biases and activation."""

    def __init__(
        self,
        n_neurons: int,
        activation: str = "relu",
        weights_initializer: str = "HeNormal",
    ):
        """Configure the layer's size, activation and weights initializer."""
        self.n_neurons = n_neurons
        self.activation_fn, self.activation_deriv = ACTIVATIONS[activation]
        self.A = None
        self.prev_A = None
        self.weights_initializer = weights_initializer
        self.W = None
        self.b = None

    def __repr__(self) -> str:
        """Return a readable summary of the layer's configuration."""
        activation_deriv_str = (
            self.activation_deriv.__name__
            if self.activation_deriv is not None
            else "None"
        )
        return (
            f"n_neurons= {self.n_neurons}, "
            f"activation_fn= {self.activation_fn.__name__}, "
            f"activation_deriv= {activation_deriv_str}, "
            f"weights_initializer= {self.weights_initializer}"
        )

    def initialize(self, n_in: int):
        """
        Initialize the layer's weights and zero biases given the input size.
        """
        initializer_fn = INITIALIZERS[self.weights_initializer]
        self.W = initializer_fn(n_in, self.n_neurons)
        self.b = np.zeros((self.n_neurons, 1))

    def forward(self):
        """
        Compute this layer's activations from the previous layer's output.
        """
        self.A = self.activation_fn(self.prev_A, self.W, self.b)

    def backward(
        self, dA: np.ndarray, learning_rate: float, output_layer: bool = False
    ) -> np.ndarray:
        """
        Compute gradients, update weights/biases,
        and return the upstream gradient.
        """
        m = self.prev_A.shape[1]

        dZ = dA if output_layer else dA * self.activation_deriv(self.A)
        dA_prev = self.W.T.dot(dZ)

        dW = (1 / m) * dZ.dot(self.prev_A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev
