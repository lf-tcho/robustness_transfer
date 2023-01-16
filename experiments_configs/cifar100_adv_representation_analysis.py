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
import foolbox as fb
from tqdm import tqdm
from ..src.models import WideResNetForLwF, WideResNetForAdvRep
from robustbench.utils import download_gdrive, rm_substr_from_state_dict, load_model
import json
from pathlib import Path
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
import os

CKPT_NAME = "ckpt"


class Cifar100RepresentationAnalysis(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        num_categories: int = 10,
        batch_size: int = 32,
        dataset_name: str = "cifar10",
        device: torch.device = torch.device("cuda"),
        epsilon=[8 / 255],
    ):
        """Initilize ImageNetExperiment.

        :param experiment_name: Name of experiment
        """
        super().__init__(experiment_name)
        self.num_categories = num_categories
        self.experiment_folder = Path("./experiments_theory") / experiment_name
        # self.model = None
        # self.model_rep = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.epsilon = epsilon

    def get_model(self):
        """Get model."""
        model = load_model(
            model_name="Addepalli2022Efficient_WRN_34_10",
            dataset="cifar100",
            threat_model="Linf",
        )
        # Change output size of model to 10 classes
        model.fc = torch.nn.Linear(640, self.num_categories)
        return model

    def get_model_rep(self):
        """Get model."""
        model_name = "Addepalli2022Efficient_WRN_34_10"
        dataset = "cifar100"
        threat_model = "Linf"
        model_dir = "./models"

        model = WideResNetForAdvRep(depth=34, widen_factor=10, num_classes=100)
        gdrive_id = '1-3c-iniqNfiwGoGPHC3nSostnG6J9fDt'
        dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
        threat_model_: ThreatModel = ThreatModel(threat_model)
        model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
        model_path = model_dir_ / f'{model_name}.pt'

        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive(gdrive_id, model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        try:
            # needed for the model of `Carmon2019Unlabeled`
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
            # needed for the model of `Chen2020Efficient`
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Change output size of model to 10 classes
        model.fc = torch.nn.Linear(640, self.num_categories)
        return model

    def run(self, device: torch.device = torch.device("cuda")):
        """Run experiment."""
        train_dataloader, eval_dataloader = self.get_dataloaders()
        # choose training or validation dataset
        data_loader = eval_dataloader

        model_rep = self.get_model_rep().to(device)
        model_rep, last_epoch, folder_ckpt = self.load_model(model_rep)
        model_rep.eval()

        feature_difference_up = 0
        feature_difference_down = 0
        count = 0

        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Set requires_grad attribute of tensor. Important for Attack
            inputs.requires_grad = True
            model_output = model_rep(inputs.to(self.device))
            clean_representation = model_output.detach().clone()
            # Calculate the loss
            loss = torch.mean(torch.norm(model_output - clean_representation ** 1.00001, dim=1))
            # Zero all existing gradients
            model_rep.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward(retain_graph=True)
            # Collect datagrad
            data_grad = inputs.grad.data
            # Call FGSM Attack
            perturbed_inputs = self.fgsm_attack(inputs, self.epsilon[0], data_grad)
            # Re-evaluate the perturbed image
            adv_output = model_rep(perturbed_inputs.to(self.device))
            feature_difference_up += torch.mean(torch.norm(adv_output - clean_representation, dim=1)).item()

            loss2 = torch.mean(torch.norm(model_output - clean_representation ** (1-0.00001), dim=1))
            # Zero all existing gradients
            model_rep.zero_grad()
            # Calculate gradients of model in backward pass
            loss2.backward()
            # Collect datagrad
            data_grad = inputs.grad.data
            # Call FGSM Attack
            perturbed_inputs = self.fgsm_attack(inputs, self.epsilon[0], data_grad)
            # Re-evaluate the perturbed image
            adv_output = model_rep(perturbed_inputs.to(self.device))
            feature_difference_down += torch.mean(torch.norm(adv_output - clean_representation, dim=1)).item()
            count += 1

        feature_difference_up /= count
        feature_difference_down /= count
        output = {"folder checkpoint": str(folder_ckpt), "feature_difference_up": feature_difference_up,
                  "feature_difference_down": feature_difference_down,
                  "epsilon": self.epsilon[0]}
        print(output)

        with open(self.experiment_folder / f"attack_on_representation_on_val_{CKPT_NAME}{last_epoch}.json", "w") as file:
            json.dump(output, file)

    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        # Return the perturbed image
        return perturbed_image


    def load_model(self, model):
        """Load latest model and get epoch."""
        ckpts = sorted(
            list(self.experiment_folder.glob("*.pth")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if ckpts:
            latest_epoch = int(ckpts[-1].stem.split("_")[-1])
            ckpt = f"{CKPT_NAME}_{latest_epoch}.pth"
            model.load_state_dict(torch.load(self.experiment_folder / ckpt))
            print(f"Model checkpoint {ckpt} loaded.")
            return model, latest_epoch, self.experiment_folder / ckpt
        else:
            return -1

    def get_dataloaders(self):
        """Get train and eval dataloader."""
        train_dataloader = get_dataloader(
            self.dataset_name,
            True,
            batch_size=self.batch_size,
            shuffle=True,
            transforms=self.transforms(),
        )
        eval_dataloader = get_dataloader(
            self.dataset_name,
            False,
            batch_size=self.batch_size,
            transforms=self.transforms()
        )
        return train_dataloader, eval_dataloader

    def transforms(self):
        """Load transforms depending on training or evaluation dataset."""
        return [Resize((32, 32)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


def main():
    """Command line tool to run experiment and evaluation."""

    experiment = Cifar100RepresentationAnalysis(
        experiment_name="bs_128_ds_intel_image_eps_20_lr_0.01_lrs_cosine_tf_method_lp",
        num_categories=6,  # cifar10=10, fashion=10, intel_image=6
        dataset_name="intel_image"
    )
    experiment.run(torch.device("cuda"))


if __name__ == "__main__":
    main()
