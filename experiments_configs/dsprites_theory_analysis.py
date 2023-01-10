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
from ..src.models import WideResNetForLwF
from robustbench.utils import download_gdrive, rm_substr_from_state_dict, load_model
import json
from pathlib import Path
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
import os

CKPT_NAME = "ckpt"


class Cifar100TheoryAnalysis(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        batch_size: int = 32,
        dataset_name: str = "dsprites",
        target_latent="orientation",
        device: torch.device = torch.device("cuda"),
        epsilon=[8 / 255],
    ):
        """Initilize ImageNetExperiment.

        :param experiment_name: Name of experiment
        """
        super().__init__(experiment_name)
        self.experiment_folder = Path("./experiments") / experiment_name
        self.ckpt_path =  f"{self.experiment_folder}/models/{target_latent}.pth"
        self.results_dir = f"{self.experiment_folder}/results"
        # self.model = None
        # self.model_rep = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.epsilon = epsilon

    def get_model(self):
        """Get model."""
        model = Model_dsprites()
        return model

    def get_model_rep(self):
        """Get model."""
        model_name = "Addepalli2022Efficient_WRN_34_10"
        dataset = "cifar100"
        threat_model = "Linf"
        model_dir = "./models"

        model = WideResNetForLwF(depth=34, widen_factor=10, num_classes=100)
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
        model = self.get_model().to(device)
        model = self.load_model(model)
        # look at the weight matrix of last linear layer
        w_matrix = model.fc.weight
        spectral_norm = torch.linalg.matrix_norm(w_matrix, 2).item()
        output = {"folder checkpoint": str(self.ckpt_path), "W_spectral_norm": spectral_norm}
        print(output)

        train_dataloader, eval_dataloader = self.get_dataloaders()
        # choose training or validation dataset
        data_loader = eval_dataloader
        model.eval()
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

        # model_rep = self.get_model_rep().to(device)
        # model_rep, _, _ = self.load_model(model_rep)
        # model_rep.eval()

        accuracy = 0
        robust_accuracy = 0
        loss = 0
        loss_adv = 0
        feature_difference = 0
        c2 = float("-inf")
        count = 0
        l_inf_pgd = fb.attacks.LinfPGD(steps=20)

        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.shape[0]
            accuracy += fb.utils.accuracy(fmodel, inputs, labels) * batch_size
            _, adv_batch, success = l_inf_pgd(fmodel, inputs, labels, epsilons=self.epsilon)
            robust_accuracy += batch_size - success.float().sum().item()
            with torch.no_grad():
                frep = model.get_features(inputs.to(self.device))
                frep_adv = model.get_features(adv_batch[0].to(self.device))
                fx = model(inputs.to(self.device))
                fx_adv = model(adv_batch[0].to(self.device))

                # model_output = model_rep(inputs.to(self.device))
                # model_output_adv = model_rep(adv_batch[0].to(self.device))
                loss += nn.MSELoss(fx, labels.to(self.device)).item()
                loss_adv += nn.MSELoss(fx_adv, labels.to(self.device)).item()
                feature_difference += torch.mean(torch.linalg.norm((frep_adv - frep), dim=1)).item()
                c2 = max(c2, nn.MSELoss(fx, labels.to(self.device)).item())
            count += 1
        loss /= count
        loss_adv /= count
        avg_feature_difference /= feature_difference/count
        effective_w_difference /= count
        effective_w_difference_on_f /= count
        accuracy = accuracy / len(data_loader.dataset)
        robust_accuracy = robust_accuracy / len(data_loader.dataset)
        output.update({"accuracy": accuracy, "robust_accuracy": robust_accuracy, "loss": loss,
                       "loss_adv": loss_adv, "avg_feature_difference": avg_feature_difference,
                       "effective_w_difference": effective_w_difference,
                       "effective_w_difference_on_f": effective_w_difference_on_f})
        print(output)

        with open(self.experiment_folder / f"weight_norms2_on_val_{CKPT_NAME}{last_epoch}.json", "w") as file:
            json.dump(output, file)

    def load_model(self, model):
        """Load latest model and get epoch."""
        model.load_state_dict(torch.load(self.ckpt_path))
        return model

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

    experiment = Cifar100TheoryAnalysis(
        experiment_name="bs_128_ds_dsprites_eps_10_lr_0.001_lrs_cosine_tf_method_lp",
        dataset_name="dsprites",
        target_latent="orientation",
    )
    experiment.run(torch.device("cuda"))


if __name__ == "__main__":
    main()
