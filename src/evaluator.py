"""Evaluate models."""

from src.experiment import Experiment
from src.trainer import CKPT_NAME
import torch
from pathlib import Path
import foolbox as fb


class Evaluator:
    def __init__(self, experiment: Experiment, epoch: int, dataloader) -> None:
        self.experiment = experiment
        self.epoch = epoch
        self.dataloader = dataloader

    def eval(self):
        """Run evaluation for an experiment."""
        model = self.load_model()
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        for inputs, labels in self.dataloader:
            _, advs, success = fb.attacks.LinfPGD()(
                fmodel, inputs, labels, epsilons=[8 / 255]
            )
            print("Robust accuracy: {:.1%}".format(1 - success.float().mean()))

    def load_model(self):
        model = self.experiment.get_model()
        model.load_state_dict(
            torch.load(
                Path("./experiments")
                / self.experiment.experiment_name
                / f"{CKPT_NAME}_{self.epoch}.pth"
            )
        )
        return model


if __name__ == "__main__":
    from experiments_configs.test_experiment import TestExperiment
    from src.dataloader import get_dataloader

    experiment = TestExperiment("test")
    dataloader = get_dataloader("cifar10", False, batch_size=1, size=5)
    evaluator = Evaluator(experiment, 4, dataloader)
    evaluator.eval()
