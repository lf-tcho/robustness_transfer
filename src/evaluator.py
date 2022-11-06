"""Evaluate models."""

from src.experiment import Experiment
from src.trainer import CKPT_NAME
import torch
from pathlib import Path
import foolbox as fb
from tqdm import tqdm


class Evaluator:
    def __init__(self, experiment: Experiment, dataloader, epoch: int = None) -> None:
        self.experiment = experiment
        self.epoch = epoch
        self.dataloader = dataloader

    def eval(self):
        """Run evaluation for an experiment."""
        model = self.load_model()
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
        accuracy = 0
        robust_accuracy = 0
        l_inf_pgd = fb.attacks.LinfPGD(steps=20)
        for inputs, labels in tqdm(self.dataloader):
            batch_size = inputs.shape[0]
            accuracy += fb.utils.accuracy(fmodel, inputs, labels) * batch_size
            _, _, success = l_inf_pgd(fmodel, inputs, labels, epsilons=[8 / 255])
            robust_accuracy += batch_size - success.float().sum().item()
            # print("Robust accuracy: {:.1%}".format(1 - success.float().mean()))
        accuracy = accuracy / len(self.dataloader.dataset)
        print(f"Accurarcy: {accuracy}")
        robust_accuracy = robust_accuracy / len(self.dataloader.dataset)
        print(f"Robust accuracy: {robust_accuracy}")

    def load_model(self):
        model = self.experiment.get_model()
        if self.epoch:
            experiment_folder = Path("./experiments") / self.experiment.experiment_name
            model.load_state_dict(
                torch.load(experiment_folder / f"{CKPT_NAME}_{self.epoch}.pth")
            )
            return model
        else:
            return model


if __name__ == "__main__":
    from experiments_configs.lp_ft_experiment import LpFtExperiment
    from src.dataloader import get_dataloader
    from torchvision.transforms import Normalize

    experiment = LpFtExperiment("lp_ft")
    transforms = [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = get_dataloader("cifar100", False, 5, 50, transforms)
    evaluator = Evaluator(experiment, dataloader)
    evaluator.eval()
