import argparse
from math import ceil
from ..src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from ..src.trainer import Trainer
from ..src.experiment import Experiment
import torch
from torchvision.transforms import Normalize, Resize
from ..src.utils import get_experiment_name
from ..src.transforms import SquarePad
from ..src.evaluator import Evaluator
from pathlib import Path
import json

CKPT_NAME = "ckpt"


class ImageNetAnalysis(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        num_categories: int = 10,
    ):
        """Initilize ImageNetExperiment.

        :param experiment_name: Name of experiment
        """
        super().__init__(experiment_name)
        self.num_categories = num_categories
        self.experiment_folder = Path("./experiments") / experiment_name
        self.model = None

    def get_model(self):
        """Get model."""
        model = load_model(
            model_name="Salman2020Do_50_2",
            dataset="imagenet",
            threat_model="Linf",
        )
        # Change output size of model to 10 classes
        model.model.fc = torch.nn.Linear(2048, self.num_categories)
        return model

    def run(self, device: torch.device = torch.device("cuda")):
        """Run experiment."""
        self.model = self.get_model().to(device)
        last_epoch, folder_ckpt = self.load_model()
        # look at the weight matrix of last linear layer
        for name, param in self.model.model.fc.named_parameters():
            if name == "weight":
                # print(name, param.shape)
                spectral_norm = torch.linalg.matrix_norm(param, 2).item()
                frobenius_norm = torch.linalg.matrix_norm(param, 'fro').item()
        output = {"folder checkpoint": str(folder_ckpt), "spectral_norm": spectral_norm, "frobenius_norm": frobenius_norm}
        print(output)
        with open(self.experiment_folder / f"weight_norms_{CKPT_NAME}{last_epoch}.json", "w") as file:
            json.dump(output, file)

    def load_model(self):
        """Load latest model and get epoch."""
        ckpts = sorted(
            list(self.experiment_folder.glob("*.pth")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if ckpts:
            latest_epoch = int(ckpts[-1].stem.split("_")[-1])
            ckpt = f"{CKPT_NAME}_{latest_epoch}.pth"
            self.model.load_state_dict(torch.load(self.experiment_folder / ckpt))
            print(f"Model checkpoint {ckpt} loaded.")
            return latest_epoch, self.experiment_folder / ckpt
        else:
            return -1


def main():
    """Command line tool to run experiment and evaluation."""
    experiment = ImageNetAnalysis(
        experiment_name="__imagenet_bs_16_ds_cifar10_eps_10_lr_0.01_lrs_None_lwf_0.1_tf_method_ft",
        num_categories=10
    )

    experiment.run(torch.device("cuda"))


if __name__ == "__main__":
    main()
