import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_metrics(experiment_path: Path):
    """Plot accuracy and robust accuracy of metric jsons.

    :param experiment_path: Path to experiment
    """
    metric_files = sorted(list(experiment_path.glob("*.json")))
    accuracy = []
    robust_accuracy = []
    epochs = []
    for metric_file in metric_files:
        num = ''.join(i for i in metric_file.stem if i.isdigit())
        if len(num) > 0:
            num = int(num)
            with open(metric_file, "r") as file:
                metrics = json.load(file)
            accuracy.append(metrics["accuracy"])
            robust_accuracy.append(metrics["robust_accuracy"])
            epochs.append(num)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epochs, accuracy, 'g-')
    ax2.plot(epochs, robust_accuracy, 'b-')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Robust accuracy', color='b')
    ax1.set_title(experiment_path.name)
    fig.savefig(experiment_path / "metrics.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder", "--folder", default=".", type=str)
    args = parser.parse_args()
    experiment_path = Path(args.folder)
    plot_metrics(experiment_path)
