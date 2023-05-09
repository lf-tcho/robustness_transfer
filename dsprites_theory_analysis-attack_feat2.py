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

class DSpritesTheoryAnalysis(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        batch_size: int = 32,
        dataset_name: str = "dsprites",
        pretrain_target_latent = "orientation",
        finetune_target_latent="scale",
        device: torch.device = torch.device("cuda"),
        attack_type="linf_pgd", 
        epsilon=[16 / 255], #[8 / 255]
    ):
        """Initilize ImageNetExperiment.

        :param experiment_name: Name of experiment
        """
        super().__init__(experiment_name)
        self.pretrain_target_latent = pretrain_target_latent
        self.finetune_target_latent = finetune_target_latent
        self.experiment_folder = Path("./experiments") / experiment_name
        self.ckpt_path =  f"{self.experiment_folder}/models/{pretrain_target_latent}-{finetune_target_latent}_{args.pretrain_model_type}_{args.model_type}.pth"
        self.results_dir = f"{self.experiment_folder}/results"
        self.attack_type = attack_type
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.epsilon = epsilon

    def get_model(self):
        """Get model."""
        model = Model_dsprites(fc_out_size=len(self.finetune_target_latent))
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
        fmodel_feat = fb.PyTorchModel(model.feats, bounds=(-1.0, 1.0))

        loss = 0
        loss_adv = 0
        loss_diff = 0
        feature_difference = 0
        c2 = float("-inf")
        count = 0
        # l_inf_pgd = fb.attacks.LinfPGD(steps=20) 
        if self.attack_type == "linf_pgd":
            attack = fb.attacks.LinfPGD(steps=20, rel_stepsize=1)
        elif self.attack_type == "l2_pgd":
            attack = fb.attacks.L2PGD(steps=50, rel_stepsize=10)
        for inputs, labels, idx in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.resize(inputs.size(0), 1, 64, 64)
            with torch.no_grad(): frep = model.get_features(inputs.to(self.device))
            batch_size = inputs.shape[0]

            criterion = fb.criteria.NormDiff(frep)
            _, adv_batch, success = attack(fmodel_feat, inputs, criterion=criterion, epsilons=self.epsilon)
            adv_batch[0] = adv_batch[0].resize(inputs.size(0), 1, 64, 64)
            with torch.no_grad():
                frep_adv = model.get_features(adv_batch[0].to(self.device))
                fx = model(inputs.to(self.device))
                fx_adv = model(adv_batch[0].to(self.device))
                loss += nn.functional.mse_loss(fx, labels.to(self.device).to(torch.float)).item()
                loss_adv += nn.functional.mse_loss(fx_adv, labels.to(self.device).to(torch.float)).item()
                loss_diff += (nn.functional.mse_loss(fx_adv, labels.to(self.device).to(torch.float)).item() - nn.functional.mse_loss(fx, labels.to(self.device).to(torch.float)).item())
                feature_difference += torch.mean(torch.linalg.norm((frep_adv - frep), dim=1)).item()
                c2 = max(c2, nn.functional.mse_loss(fx, labels.to(self.device).to(torch.float)).item())
            count += 1
        loss /= count
        loss_adv /= count
        loss_diff /= count
        avg_feature_difference = feature_difference/count
        
        lhs = loss_diff/spectral_norm
        rhs = avg_feature_difference
        output.update({"loss": loss, "loss_adv": loss_adv, "avg_feature_difference": avg_feature_difference,
                       "C2": c2, "loss_diff": loss_diff, "LHS": lhs, "RHS": rhs})
        print(output)
        
        with open(self.experiment_folder / f"result_th3_2_attack_feat2_{args.pretrain_target_latent}-{self.finetune_target_latent}_{args.pretrain_model_type}_{args.model_type}_{self.attack_type}_reg_on_val.json", "w") as file:
            json.dump(output, file)

    def load_model(self, model):
        """Load latest model and get epoch."""
        model.load_state_dict(torch.load(self.ckpt_path))
        return model

    def get_dataloaders(self):
        """Get train and eval dataloader."""
        train_dataloader = get_dataloader(
            self.finetune_target_latent,
            self.dataset_name,
            True,
            batch_size=4096,
            shuffle=True,
        )
        eval_dataloader = get_dataloader(
            self.finetune_target_latent,
            self.dataset_name,
            False,
            batch_size=4096,
        )
        return train_dataloader, eval_dataloader


def main(args):
    """Command line tool to run experiment and evaluation."""

    experiment = DSpritesTheoryAnalysis(
        experiment_name=args.exp_name,
        dataset_name=args.dataset_name,
        pretrain_target_latent=args.pretrain_target_latent,
        finetune_target_latent=args.finetune_target_latent,
        attack_type=args.attack_type 
    )
    experiment.run(torch.device("cuda"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="bs_128_ds_dsprites_eps_10_lr_0.001_lrs_cosine_tf_method_lp",
            help="name od the experiment/save dir")
    parser.add_argument('--dataset_name', type=str, default="dsprites",
            help="name of the dataset")
    parser.add_argument('--pretrain_model_type', type=str, default="robust", choices=["robust", "clean", "random"])
    parser.add_argument('--pretrain_target_latent', type=str, default="orientation")
    parser.add_argument('--finetune_target_latent', type=str, default="scale")
    parser.add_argument('--attack_type', type=str, default="linf_pgd")
    parser.add_argument('--model_type', type=str, choices=["lp", "ft"], default="lp")
    args = parser.parse_args()
    args.pretrain_target_latent = args.pretrain_target_latent.split(', ')
    args.finetune_target_latent = args.finetune_target_latent.split(', ')

    main(args)
