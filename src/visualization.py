import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from src.model.network import Network


def _plot_loss(ax: Axes, network: Network):
    ax.plot(
        range(len(network.loss_train)),
        network.loss_train,
        c="blue",
        label="training loss"
    )
    ax.plot(
        range(len(network.loss_valid)),
        network.loss_valid,
        linestyle="-",
        c="orange",
        label="validation loss"
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()


def _plot_accuracy(ax: Axes, network: Network):
    ax.plot(
        range(len(network.acc_train)),
        network.acc_train,
        c="blue",
        label="training acc"
    )
    ax.plot(
        range(len(network.acc_valid)),
        network.acc_valid,
        c="orange",
        label="validation acc"
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()


def visualization(network: Network):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    _plot_loss(axes[0], network)
    _plot_accuracy(axes[1], network)

    plt.tight_layout()
    plt.show()
