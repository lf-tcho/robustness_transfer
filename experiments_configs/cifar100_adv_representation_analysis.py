import argparse
from math import ceil
from ..src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
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
        attack_type="l2_pgd"
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
        self.attack_type = attack_type

    def get_model(self):
        """Get model."""
        model = load_model(
            model_name="Addepalli2022Efficient_WRN_34_10",
            dataset="cifar100",
            threat_model="Linf",
        )
        # Change output size of model to 10 classes
        if self.num_categories > 0:
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
        if self.num_categories > 0:
            model.fc = torch.nn.Linear(640, self.num_categories)
        return model

    def run(self, device: torch.device = torch.device("cuda")):
        """Run experiment."""
        eval_dataloader = self.get_dataloaders(train=False)
        # choose training or validation dataset
        data_loader = eval_dataloader

        model_rep = self.get_model_rep().to(device)
        model_rep, last_epoch, folder_ckpt = self.load_model(model_rep)
        model_rep.eval()

        feature_difference = 0
        count = 0

        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            model_output = model_rep(inputs.to(self.device))
            clean_representation = model_output.detach().clone()

            # Call FGSM Attack
            if self.attack_type == "fgsm":
                perturbed_inputs = self.fgsm_attack(model_rep, inputs, self.epsilon[0])
            elif self.attack_type == "linf_pgd":
                perturbed_inputs = self.linf_pgd_attack(model_rep, inputs, self.epsilon[0], num_steps=10, step_size=0.01)
            elif self.attack_type == "l2_pgd":
                perturbed_inputs = self.l2_pgd_attack(model_rep, inputs, self.epsilon[0], num_steps=10, step_size=0.01)

            # Re-evaluate the perturbed image
            adv_output = model_rep(perturbed_inputs.to(self.device))
            feature_difference += torch.mean(torch.norm(adv_output - clean_representation, dim=1)).item()
            count += 1

        feature_difference /= count
        output = {"folder checkpoint": str(folder_ckpt), "feature_difference": feature_difference,
                  "epsilon": self.epsilon[0]}
        print(output)

        with open(self.experiment_folder / f"attack_{self.attack_type}_on_representation_on_val_{CKPT_NAME}{last_epoch}_{self.dataset_name}.json", "w") as file:
            json.dump(output, file)

    # FGSM attack code
    def fgsm_attack(self, model_rep, image, epsilon):
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
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [-1,1] range
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        # Return the perturbed image
        return perturbed_image

    # LinfPGD Attack
    def linf_pgd_attack(self, model, image, epsilon, num_steps=10, step_size=0.01):
        original_output = model(image.to(self.device))
        perturbed_image = image.detach().clone().to(self.device)
        # l-inf specific random start
        perturbed_image = perturbed_image + (-2*epsilon)*torch.rand(perturbed_image.shape).to(self.device) + epsilon
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        for i in range(num_steps):
            perturbed_image.requires_grad = True 
            model.zero_grad()
            perturbed_output = model(perturbed_image.to(self.device))
            loss = torch.mean(torch.norm(perturbed_output - original_output, dim=1))  # F.mse_loss(perturbed_output, original_output, reduction="sum")
            loss.backward(retain_graph=True)
            # l-inf specific update step
            perturbed_image = perturbed_image + torch.clamp(step_size * perturbed_image.grad.data.sign(), -epsilon, epsilon)
            # limit the perturbation
            # perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
            perturbed_image = perturbed_image.detach()
            # Adding clipping to maintain [-1,1] range
            perturbed_image = torch.clamp(perturbed_image, -1, 1)
        return perturbed_image

    # L2 PGD Attack
    def l2_pgd_attack(self, model, image, epsilon, num_steps=10, step_size=0.01):
        original_output = model(image.to(self.device))
        perturbed_image = image.detach().clone().to(self.device)
        # random start
        batch_size, n = torch.flatten(perturbed_image, start_dim=1).shape
        x = torch.normal(mean=torch.zeros(batch_size, n+1), std=1).to(self.device)
        nr = torch.linalg.norm(x, dim=1, keepdims=True)
        s = x/nr
        r = s[:, :n].reshape(perturbed_image.shape)
        perturbed_image = perturbed_image + epsilon * r
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        for i in range(num_steps):
            perturbed_image.requires_grad = True
            model.zero_grad()
            perturbed_output = model(perturbed_image.to(self.device))
            loss = torch.mean(torch.norm(perturbed_output - original_output,
                                         dim=1))
            loss.backward(retain_graph=True)
            # update step, normalize gradient
            norms = torch.linalg.norm(torch.flatten(perturbed_image.grad.data, start_dim=1), dim=1)
            factor = 1 / torch.maximum(norms, torch.tensor((1e-12)))
            factor_reshape = factor.shape + (1,) * (perturbed_image.grad.data.ndim - factor.ndim)
            factor = factor.reshape(factor_reshape)
            normalized_gradient = perturbed_image.grad.data * factor
            projected_update = step_size * normalized_gradient
            # limit the perturbation, project
            norms2 = torch.linalg.norm(torch.flatten(projected_update, start_dim=1), dim=1)
            norms2 = torch.maximum(norms2, torch.tensor((1e-12)))
            factor = torch.minimum( torch.tensor((1)), epsilon/norms2)
            factor_reshape = factor.shape + (1,) * (projected_update.ndim - factor.ndim)
            factor = factor.reshape(factor_reshape)
            perturbed_image = perturbed_image + projected_update * factor
            perturbed_image = perturbed_image.detach()
            # Adding clipping to maintain [-1,1] range
            perturbed_image = torch.clamp(perturbed_image, -1, 1)
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
            print(f"No model checkpoint loaded.")
            return model, -1, self.experiment_folder

    def get_dataloaders(self, train=True):
        """Get train and eval dataloader."""
        eval_dataloader = get_dataloader(
            self.dataset_name,
            False,
            batch_size=self.batch_size,
            transforms=self.transforms()
        )
        if train:
            train_dataloader = get_dataloader(
                self.dataset_name,
                True,
                batch_size=self.batch_size,
                shuffle=True,
                transforms=self.transforms(),
            )
            return train_dataloader, eval_dataloader
        return eval_dataloader

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
