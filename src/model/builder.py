from src.model.network import Network
from src.model.layer import DenseLayer


def network_builder(
        layers: tuple[int],
        activation: str,
        weights_initializer: str = "HeNormal"
) -> Network:
    layers_config = tuple(
        # Hidden Layers
        DenseLayer(n_neurons, activation, weights_initializer)
        for n_neurons in layers[:-1]
        # Output Layer
    ) + (DenseLayer(layers[-1], "softmax", "GlorotNormal"),)

    return Network(layers_config)
