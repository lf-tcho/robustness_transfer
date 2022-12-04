import argparse
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json


def plot_metrics(experiment_path: Path):
    """Plot accuracy and robust accuracy of metric jsons.

    :param experiment_path: Path to experiment
    """
    matplotlib.rcParams.update({'font.size': 18})
    metric_files = sorted(list(experiment_path.glob("*.json")))
    accuracy = []
    robust_accuracy = []
    for metric_file in metric_files:
        num = ''.join(i for i in metric_file.stem if i.isdigit())
        if len(num) > 0:
            num = int(num)
            with open(metric_file, "r") as file:
                metrics = json.load(file)
            accuracy.append((num, metrics["accuracy"]))
            robust_accuracy.append((num, metrics["robust_accuracy"]))
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 6)
    ax2 = ax1.twinx()
    accuracy = sorted(accuracy)
    robust_accuracy = sorted(robust_accuracy)
    lns1 = ax1.plot([i[0] for i in accuracy], [i[1] for i in accuracy], 'g-', label="Accuracy")
    lns2 = ax2.plot([i[0] for i in robust_accuracy], [i[1] for i in robust_accuracy], 'b-', label="Robust accuracy")
    lns3 = plt.axvspan(0, 19, facecolor='r', alpha=0.2, label="Linear probing")
    lns4 = plt.axvspan(19, 29, facecolor='y', alpha=0.2, label="Full fine-tuning")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Robust accuracy')
    lns = lns1+lns2+[lns3]+[lns4]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center left")
    ax1.set_title("Linear probing then full fine-tuning")
    fig.savefig(experiment_path / "metrics.png")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-folder", "--folder", default=".", type=str)
    # args = parser.parse_args()
    # experiment_path = Path(args.folder)
    experiment_path = Path(r"C:\Users\mauri\Documents\Stanford\dmtml\project\dmtml\tmp\plot_lpft")
    plot_metrics(experiment_path)
