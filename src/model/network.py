import numpy as np
from tqdm import tqdm

from src.constants import BATCH_SIZE
from src.model.layer import DenseLayer
from src.model.loss import binary_cross_entropy
from src.model.metrics import ft_accuracy, ft_f1_score, ft_precision, ft_recall


class EarlyStopping:
    """Tracks validation loss to decide when training should stop."""

    def __init__(self, patience: int):
        """Initialize the patience counter and best loss seen so far."""
        self.patience = patience
        self.counter = patience
        self.best_loss = float("inf")

    def should_stop(self):
        """Return True once the patience counter has been exhausted."""
        return self.counter == 0

    def update_loss(self, loss):
        """
        Update the best loss and patience counter; return True on improvement.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = self.patience
            return True

        self.counter -= 1
        return False


class Network:
    """
    A feedforward neural network made of DenseLayers,
    trained by backpropagation.
    """

    def __init__(self, layers: tuple[DenseLayer, ...]):
        """Store the layers and initialize training history containers."""
        self.layers = layers
        self.params = {}
        self.loss_train = []
        self.loss_valid = []
        self.acc_train = []
        self.acc_valid = []
        self.precision_train = []
        self.precision_valid = []
        self.recall_train = []
        self.recall_valid = []
        self.f1_score_train = []
        self.f1_score_valid = []
        self.stopping_reason = ""
        self.best_epoch = 0

    def __repr__(self):
        """Return the layers' representation."""
        return f"{self.layers}"

    def _initialize(self, n_in: int):
        """
        Initialize weights and biases of every layer given the input size.
        """
        for layer in self.layers:
            layer.initialize(n_in)
            n_in = layer.n_neurons

    def load_parameters(self, W: tuple[np.ndarray], b: tuple[np.ndarray]):
        """Assign pre-trained weights and biases to each layer, in order."""
        for layer_index, layer in enumerate(self.layers):
            layer.W = W[layer_index]
            layer.b = b[layer_index]

    def save_parameters(self, params_path: str):
        """Save the best recorded weights and biases to an .npz file."""
        np.savez(params_path, **self.params)
        print(f"Weights and biases saved to '{params_path}'")

    def _update_best_parameters(self):
        """Snapshot each layer's current weights and biases as the new best."""
        for layer_index, layer in enumerate(self.layers, start=1):
            self.params[f"W{layer_index}"] = np.copy(layer.W)
            self.params[f"b{layer_index}"] = np.copy(layer.b)

    def _restore_best_parameters(self):
        """
        Restore each layer's weights and biases from the last best snapshot.
        """
        for layer_index, layer in enumerate(self.layers, start=1):
            layer.W = np.copy(self.params[f"W{layer_index}"])
            layer.b = np.copy(self.params[f"b{layer_index}"])

    def _truncate_history_to_best_epoch(self):
        best_epoch = self.best_epoch
        # Train
        self.loss_train = self.loss_train[:best_epoch]
        self.acc_train = self.acc_train[:best_epoch]
        self.precision_train = self.precision_train[:best_epoch]
        self.recall_train = self.recall_train[:best_epoch]
        self.f1_score_train = self.f1_score_train[:best_epoch]
        # Valid
        self.loss_valid = self.loss_valid[:best_epoch]
        self.acc_valid = self.acc_valid[:best_epoch]
        self.precision_valid = self.precision_valid[:best_epoch]
        self.recall_valid = self.recall_valid[:best_epoch]
        self.f1_score_valid = self.f1_score_valid[:best_epoch]

    def forward(self, X_train: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through all layers and
        return the output activations.
        """
        prev_A = X_train
        for layer in self.layers:
            layer.prev_A = prev_A
            layer.forward()
            prev_A = layer.A

        return prev_A

    def backward(
        self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float
    ):
        """
        Run a backward pass through all layers, updating their parameters.
        """
        dZ = y_pred - y_true
        dA = self.layers[-1].backward(dZ, learning_rate, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA, learning_rate)

    def _evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        mode: str,
        y_pred: np.ndarray | None = None,
    ) -> float:
        """
        Compute loss and metrics for a set,
        appending them to the train/valid history.
        """
        y_pred = self.forward(X) if y_pred is None else y_pred

        loss = binary_cross_entropy(y_true, y_pred)
        accuracy = ft_accuracy(y_true, y_pred)
        precision = ft_precision(y_true, y_pred)
        recall = ft_recall(y_true, y_pred)
        f1_score = ft_f1_score(y_true, y_pred)

        if mode == "train":
            self.loss_train.append(loss)
            self.acc_train.append(accuracy)
            self.precision_train.append(precision)
            self.recall_train.append(recall)
            self.f1_score_train.append(f1_score)
        elif mode == "valid":
            self.loss_valid.append(loss)
            self.acc_valid.append(accuracy)
            self.precision_valid.append(precision)
            self.recall_valid.append(recall)
            self.f1_score_valid.append(f1_score)

        return loss

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        epochs: int,
        disable_epochs_metrics: bool,
        learning_rate: float,
        patience: int,
    ):
        """
        Train the network with gradient descent
        until max epochs or early stopping.
        """
        self._initialize(X_train.shape[0])

        early_stopping = EarlyStopping(patience)
        epochs_iter = (
            tqdm(range(1, epochs + 1))
            if disable_epochs_metrics
            else range(1, epochs + 1)
        )

        epoch = 0
        for epoch in epochs_iter:
            # Shuffle
            permuted_indexes = np.random.permutation(len(X_train.T))
            X_train = X_train[:, permuted_indexes]
            y_train = y_train[:, permuted_indexes]
            # Mini-batch
            for start in range(0, len(X_train), BATCH_SIZE):
                X_train_batch = X_train[:, start : start + BATCH_SIZE]
                y_train_batch = y_train[:, start : start + BATCH_SIZE]

                # Forward Propagation
                y_pred = self.forward(X_train_batch)

                # Backward Propagation
                self.backward(y_train_batch, y_pred, learning_rate)

            # Evaluate Loss + Accuracy
            y_pred = self.forward(X_train)
            loss_train = self._evaluate(X_train, y_train, "train", y_pred)
            loss_valid = self._evaluate(X_valid, y_valid, "valid")

            # Display Epochs Metrics
            if disable_epochs_metrics is False:
                print(
                    f"epoch {epoch}/{epochs} -",
                    f"loss: {loss_train:.4f} - val_loss: {loss_valid:.4f}",
                )

            # Early Stopping
            if early_stopping.update_loss(loss_valid):
                self._update_best_parameters()
                self.best_epoch = epoch
            if early_stopping.should_stop():
                break

        self._restore_best_parameters()
        self._truncate_history_to_best_epoch()

        self.stopping_reason = (
            f"early stopping (epoch: {epoch})"
            if early_stopping.counter == 0
            else f"max epochs reached: {epochs}"
        )

    def predict(self, X) -> np.ndarray:
        """
        Return the network's output activations
        (class probabilities) for X.
        """
        return self.forward(X)
