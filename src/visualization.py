import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.model.network import Network


def _plot_metric(
        ax: Axes,
        metric_train: list,
        metric_valid: list,
        metric: str
):
    ax.plot(
        range(len(metric_train)),
        metric_train,
        c="blue",
        label="training"
    )
    ax.plot(
        range(len(metric_valid)),
        metric_valid,
        linestyle="-",
        c="orange",
        label="validation"
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.set_title(metric.capitalize())


def visualization(network: Network):
    _, axes = plt.subplots(2, 3, figsize=(16, 6))

    _plot_metric(axes[0][0], network.loss_train, network.loss_valid, "loss")
    _plot_metric(axes[0][1], network.acc_train, network.acc_valid, "accuracy")
    _plot_metric(
        axes[1][0],
        network.precision_train,
        network.precision_valid,
        "precision"
    )
    _plot_metric(
        axes[1][1],
        network.recall_train,
        network.recall_valid,
        "recall"
    )
    _plot_metric(
        axes[1][2],
        network.f1_score_train,
        network.f1_score_valid,
        "f1-score"
    )

    axes[0][2].set_visible(False)
    plt.tight_layout()
    plt.show()
