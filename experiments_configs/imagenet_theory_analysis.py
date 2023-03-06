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
import numpy as np

from torchvision.transforms import Normalize, Resize
import foolbox as fb
from tqdm import tqdm
from torchvision.models.resnet import Bottleneck
from ..src.models import WideResNetForLwFImageNet
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.utils import download_gdrive, rm_substr_from_state_dict, load_model, add_substr_to_state_dict
import json
from pathlib import Path
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
import os

CKPT_NAME = "ckpt"


class ImageNetTheoryAnalysis(Experiment):
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
            model_name="Salman2020Do_50_2",
            dataset="imagenet",
            threat_model="Linf",
        )
        # Change output size of model to 10 classes
        if self.num_categories > 0:
            model.model.fc = torch.nn.Linear(2048, self.num_categories)
        return model

    def get_model_rep(self):
        """Get model."""
        model_name = "Salman2020Do_50_2"
        dataset = "imagenet"
        threat_model = "Linf"
        model_dir = "./models"

        # ('Salman2020Do_50_2', {
        #     'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
        #     'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
        #     'preprocessing': 'Res256Crop224'
        # }),
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)
        kwargs = dict()
        kwargs['width_per_group'] = 64 * 2
        model = WideResNetForLwFImageNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)  # WideResNetForLwFImageNet

        model = normalize_model(model, mu,
                                sigma)  # normalize_model(pt_models.wide_resnet50_2(), mu, sigma) # WideResNetForLwF(depth=34, widen_factor=10, num_classes=100)
        gdrive_id = '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB'
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

        state_dict = add_substr_to_state_dict(state_dict, 'model.')

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Change output size of model to num_category classes
        if self.num_categories > 0:
            model.model.fc = torch.nn.Linear(2048, self.num_categories)
        return model

    def run(self, device: torch.device = torch.device("cuda")):
        """Run experiment."""
        model = self.get_model().to(device)
        model, last_epoch, folder_ckpt = self.load_model(model)
        # look at the weight matrix of last linear layer
        w_matrix = model.model.fc.weight
        mean_max_dif = 0
        max_dif_list = []
        wi_norm_list = []
        for i in range(w_matrix.shape[0]):
            max_dif = 0
            temp_list = []
            for j in range(w_matrix.shape[0]):
                temp = torch.norm(w_matrix[j, :] - w_matrix[i, :], 'fro').item()
                temp_list.append(temp)
                if temp > max_dif:
                    max_dif = temp
            #print("diff from W_", i, ":", temp_list)
            wi_norm_list.append(torch.norm(w_matrix[i, :]).item())
            max_dif_list.append(max_dif)
            mean_max_dif += max_dif
        #print("norm of all  W_i:", wi_norm_list)
        mean_max_dif /= w_matrix.shape[0]
        spectral_norm = torch.linalg.matrix_norm(w_matrix, 2).item()
        frobenius_norm = torch.linalg.matrix_norm(w_matrix, 'fro').item()
        output = {"folder checkpoint": str(folder_ckpt), "spectral_norm": spectral_norm,
                  "frobenius_norm": frobenius_norm, "mean_L2_norm_Wi": sum(wi_norm_list)/len(wi_norm_list),
                  "Mean_Max_dif": mean_max_dif,
                  "Max_Max_dif": max(max_dif_list),
                  "Min_Max_dif": min(max_dif_list)}
        print(output)
        # with open(self.experiment_folder / f"theory_linear_layer constants_{CKPT_NAME}{last_epoch}.json", "w") as file:
        #     json.dump(output, file)

        eval_dataloader = self.get_dataloaders(train=False)
        # choose training or validation dataset
        data_loader = eval_dataloader
        model.eval()
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

        model_rep = self.get_model_rep().to(device)
        model_rep, _, _ = self.load_model(model_rep)
        model_rep.eval()

        accuracy = 0
        robust_accuracy = 0
        cross_entropy = 0
        cross_entropy_adv = 0
        feature_difference = 0
        relative_feature_difference = 0
        effective_w_difference = 0
        effective_w_difference_on_f = 0
        softmax_feature_diff = 0
        thm_41_dif = 0
        thm_41_acc = 0
        thm_41_contradiction = 0
        count = 0
        l_inf_pgd = fb.attacks.LinfPGD(steps=20)

        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.shape[0]
            accuracy += fb.utils.accuracy(fmodel, inputs, labels) * batch_size
            _, adv_batch, success = l_inf_pgd(fmodel, inputs, labels, epsilons=self.epsilon)
            robust_accuracy += batch_size - success.float().sum().item()
            with torch.no_grad():
                model_output = model_rep(inputs.to(self.device))
                model_output_adv = model_rep(adv_batch[0].to(self.device))
                cross_entropy += nn.CrossEntropyLoss()(model_output[0], labels.to(self.device)).item()
                cross_entropy_adv += nn.CrossEntropyLoss()(model_output_adv[0], labels.to(self.device)).item()
                feature_difference += torch.mean(torch.norm(model_output[1] - model_output_adv[1], dim=1)).item()
                relative_feature_difference += torch.mean(
                    torch.norm(model_output[1] - model_output_adv[1], dim=1) / torch.norm(model_output[1], dim=1)
                ).item()
                for i in range(batch_size):
                    min_lhs = np.inf
                    for j in range(self.num_categories):
                        if j != labels[i]:
                            temp_lhs = torch.norm(model_output[0][i, labels[i]] - model_output[0][i, j]) \
                                       / torch.norm(w_matrix[labels[i], :] - w_matrix[j, :])
                            if temp_lhs < min_lhs:
                                min_lhs = temp_lhs
                    temp_dif = min_lhs.item() - torch.mean(
                        torch.norm(model_output[1] - model_output_adv[1], dim=1)).item()
                    thm_41_dif += temp_dif / batch_size
                    temp_acc = 1 if (temp_dif > 0 and torch.argmax(model_output[0][i, :]) == labels[i]) else 0
                    thm_41_acc += temp_acc / batch_size
                    temp_contra = 1 if (temp_dif >= 0 and torch.argmax(model_output_adv[0][i, :]) != labels[i] and
                                        torch.argmax(model_output[0][i, :]) == labels[i]) else 0
                    if temp_contra == 1:
                        print("---------------------------------")
                        print(labels[i], model_output_adv[0][i, :])
                        print(temp_dif, min_lhs.item(),
                              torch.mean(torch.norm(model_output[1] - model_output_adv[1], dim=1)).item())
                        print("---------------------------------")
                    thm_41_contradiction += temp_contra / batch_size
                #     effective_w_difference += torch.norm(
                #             w_matrix[labels[i], :] -
                #             torch.matmul(
                #                 torch.nn.functional.softmax(model_output_adv[0], dim=1)[i, :],
                #                 w_matrix
                #             )
                #     ).item() / batch_size
                #     effective_w_difference_on_f += torch.norm(
                #         w_matrix[labels[i], :] -
                #         torch.matmul(
                #             torch.nn.functional.softmax(model_output_adv[0], dim=1)[i, :],
                #             w_matrix
                #         )
                #     ).item() * torch.norm(model_output[1] - model_output_adv[1], dim=1)[i].item() / batch_size
                #     softmax_feature_diff += (
                #         1-torch.nn.functional.softmax(model_output_adv[0], dim=1)[i, labels[i]].item()
                #     ) * torch.norm(model_output[1] - model_output_adv[1], dim=1)[i].item() / batch_size
            count += 1
        cross_entropy /= count
        cross_entropy_adv /= count
        feature_difference /= count
        relative_feature_difference /= count
        effective_w_difference /= count
        effective_w_difference_on_f /= count
        softmax_feature_diff /= count
        thm_41_dif /= count
        thm_41_acc /= count
        thm_41_contradiction /= count
        accuracy = accuracy / len(data_loader.dataset)
        robust_accuracy = robust_accuracy / len(data_loader.dataset)
        output.update({"accuracy": accuracy, "robust_accuracy": robust_accuracy, "cross_entropy": cross_entropy,
                       "cross_entropy_adv": cross_entropy_adv, "feature_difference": feature_difference,
                       "relative_feature_difference": relative_feature_difference,
                       "thm_41_dif": thm_41_dif,
                       "thm_41_acc": thm_41_acc,
                       "thm_41_contradiction": thm_41_contradiction,
                       # "effective_w_difference": effective_w_difference,
                       # "effective_w_difference_on_f": effective_w_difference_on_f,
                       # "softmax_feature_diff": softmax_feature_diff}
                       })
        print(output)

        with open(
                self.experiment_folder / f"theory_constants_on_val_with_thm4_{CKPT_NAME}{last_epoch}_{self.dataset_name}.json",
                "w") as file:
            json.dump(output, file)

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
        return [Resize((256, 256))]


def main():
    """Command line tool to run experiment and evaluation."""

    experiment = ImageNetTheoryAnalysis(
        experiment_name="__imagenet_bs_32_ds_fashion_eps_10_lr_0.001_lrs_cosine_tf_method_lp",
        num_categories=10,  # cifar10=10; 1/255, fashion=10; 1/255, intel_image=6; 4/255 also image_net
        dataset_name="fashion",
        epsilon=[1/255],  # instead of 8/255
        batch_size=32
    )
    experiment.run(torch.device("cuda"))


if __name__ == "__main__":
    main()
