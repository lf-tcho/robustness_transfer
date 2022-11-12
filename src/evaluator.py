"""Evaluate models."""

from src.experiment import Experiment
from src.trainer import CKPT_NAME
import torch
from pathlib import Path
import foolbox as fb
from tqdm import tqdm


class Evaluator:
    """Class to evaluate metrics (e.g. accurarcy or robost accuracy) for a given experiment."""

    def __init__(
        self,
        experiment: Experiment,
        dataloader,
        epoch: int = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.experiment = experiment
        self.epoch = epoch
        self.dataloader = dataloader
        self.device = device

    def eval(self):
        """Run evaluation for an experiment."""
        model = self.load_model()
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
        accuracy = 0
        robust_accuracy = 0
        l_inf_pgd = fb.attacks.LinfPGD(steps=20)
        for inputs, labels in tqdm(self.dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.shape[0]
            accuracy += fb.utils.accuracy(fmodel, inputs, labels) * batch_size
            _, _, success = l_inf_pgd(fmodel, inputs, labels, epsilons=[8 / 255])
            robust_accuracy += batch_size - success.float().sum().item()
        accuracy = accuracy / len(self.dataloader.dataset)
        print(f"Accurarcy: {accuracy}")
        robust_accuracy = robust_accuracy / len(self.dataloader.dataset)
        print(f"Robust accuracy: {robust_accuracy}")

    def load_model(self):
        """Load model.

        If self.epoch is None loads original model (no checkpoint).
        If self.epoch is given load respective checkpoint.
        """
        model = self.experiment.get_model().to(self.device)
        if self.epoch:
            experiment_folder = Path("./experiments") / self.experiment.experiment_name
            ckpt = experiment_folder / f"{CKPT_NAME}_{self.epoch}.pth"
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f"Loaded checkpoint {ckpt}")
            return model
        else:
            return model


if __name__ == "__main__":
    from experiments_configs.lp_experiment import LpExperiment
    from src.dataloader import get_dataloader
    from torchvision.transforms import Normalize

    experiment = LpExperiment("lp_bs_128_eps_10_lr_0.001")
    transforms = [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = get_dataloader("cifar10", False, 5, 50, transforms)
    evaluator = Evaluator(experiment, dataloader, epoch=9)
    evaluator.eval()
