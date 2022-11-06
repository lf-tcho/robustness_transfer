from src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
import torch.nn as nn
from src.trainer import Trainer
from src.experiment import Experiment
import torch


class LpFtExperiment(Experiment):
    """Experiment for linear probing then fine tuning."""

    def get_model(self):
        """Get model."""
        return load_model(
            model_name="Addepalli2022Efficient_WRN_34_10",
            dataset="cifar100",
            threat_model="Linf",
        )

    def run(self):
        """Run experiment."""
        model = self.get_model()
        model.fc = torch.nn.Linear(640, 10)
        train_dataloader = get_dataloader("cifar10", True, batch_size=1, size=10)
        eval_dataloader = get_dataloader("cifar10", False, batch_size=1, size=5)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        trainer = Trainer(
            model,
            train_dataloader,
            eval_dataloader,
            loss,
            5,
            optimizer,
            self.experiment_name,
        )
        trainer.train()


if __name__ == "__main__":
    experiment = LpFtExperiment("lp_ft")
    experiment.run()
