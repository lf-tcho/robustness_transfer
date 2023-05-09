import argparse
from math import ceil
from src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from src.trainer import Trainer
from src.experiment import Experiment
import torch
import torch.nn.functional as F

from torchvision.transforms import Normalize, Resize
import foolbox as fb
from tqdm import tqdm
from src.models import *
from robustbench.utils import download_gdrive, rm_substr_from_state_dict, load_model
import json
from pathlib import Path
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
import os

CKPT_NAME = "ckpt"

class TheoryAnalysis(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        batch_size: int = 32,
        dataset_name: str = "dsprites",
        device: torch.device = torch.device("cuda"),
        epsilon=[8 / 255],
    ):
        """Initilize ImageNetExperiment.

        :param experiment_name: Name of experiment
        """
        super().__init__(experiment_name)
        self.experiment_folder = Path("./experiments") / experiment_name
        self.ckpt_path =  f"{self.experiment_folder}/models/cifar100-cifar10_{args.model_type}.pth"
        self.results_dir = f"{self.experiment_folder}/results"
        # self.model = None
        # self.model_rep = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.epsilon = epsilon

    def get_model(self, model_name="Addepalli2022Efficient_WRN_34_10"):
        """Get model."""
        from robustbench.utils import load_model
        model = load_model(
            model_name=model_name,
            dataset="cifar100",
            threat_model="Linf",
        )
        # Change output size of model to 10 classes
        model.fc = torch.nn.Linear(640, 10)
        return model

    def run(self, device: torch.device = torch.device("cuda")):
        """Run experiment."""
        model = self.get_model().to(device)
        model = self.load_model(model)
        # feature extractor model
        modules = list(model.children())[:-1]
        model_feat = nn.Sequential(*modules)
        
        # look at the weight matrix of last linear layer
        w_matrix = model.fc.weight
        spectral_norm = torch.linalg.matrix_norm(w_matrix, 2).item()
        output = {"folder checkpoint": str(self.ckpt_path), "W_spectral_norm": spectral_norm}
        print(output)

        train_dataloader, eval_dataloader = self.get_dataloaders()
        # choose training or validation dataset
        data_loader = eval_dataloader
        model.eval()
        fmodel_feat = fb.PyTorchModel(model_feat, bounds=(-1.0, 1.0))

        loss = 0
        loss_adv = 0
        feature_difference = 0
        c2 = float("-inf")
        count = 0
        l_inf_pgd = fb.attacks.LinfPGD(steps=20) 
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.resize(inputs.size(0), 3, 32, 32)
            with torch.no_grad(): frep = model_feat(inputs.to(self.device))
            batch_size = inputs.shape[0]

            criterion = fb.criteria.NormDiff(frep)
            _, adv_batch, success = l_inf_pgd(fmodel_feat, inputs, criterion=criterion, epsilons=self.epsilon)
            adv_batch[0] = adv_batch[0].resize(inputs.size(0), 3, 32, 32)
            with torch.no_grad():
                frep_adv = model_feat(adv_batch[0].to(self.device))
                fx = model(inputs.to(self.device))
                fx_adv = model(adv_batch[0].to(self.device))
                loss += nn.functional.cross_entropy(fx, labels.to(self.device)).item()
                loss_adv += nn.functional.cross_entropy(fx_adv, labels.to(self.device)).item()
                feature_difference += torch.mean(torch.linalg.norm((frep_adv - frep), dim=1)).item()
                c2 = max(c2, nn.functional.cross_entropy(fx, labels.to(self.device)).item())
            count += 1
        loss /= count
        loss_adv /= count
        avg_feature_difference = feature_difference/count
        output.update({"loss": loss, "loss_adv": loss_adv, "avg_feature_difference": avg_feature_difference,
                       "C2": c2})
        print(output)
        
        with open(self.experiment_folder / f"result_th3_2_attack_feat2_cifar100-cifar10_{args.model_type}_on_val.json", "w") as file:
            json.dump(output, file)

    def load_model(self, model):
        """Load latest model and get epoch."""
        model.load_state_dict(torch.load(self.ckpt_path))
        return model

    def get_dataloaders(self):
        """Get train and eval dataloader."""
        train_dataloader = get_dataloader(
            '',
            self.dataset_name,
            True,
            batch_size=1,
            shuffle=True,
        )
        eval_dataloader = get_dataloader(
            '',
            self.dataset_name,
            False,
            batch_size=1,
        )
        return train_dataloader, eval_dataloader


def main(args):
    """Command line tool to run experiment and evaluation."""

    experiment = TheoryAnalysis(
        experiment_name=args.exp_name,
        dataset_name=args.dataset_name,
    )
    experiment.run(torch.device("cuda"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="bs_128_ds_cifar100-cifar10_eps_10_lr_0.001_lrs_cosine_tf_method_lp",
            help="name od the experiment/save dir")
    parser.add_argument('--dataset_name', type=str, default="cifar10",
            help="name of the dataset")
    parser.add_argument('--model_type', type=str, choices=["lp", "ft"], default="lp")
    args = parser.parse_args()

    main(args)