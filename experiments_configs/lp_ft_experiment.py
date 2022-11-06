from src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
import torch.nn as nn
from src.trainer import Trainer
from src.experiment import Experiment
import torch
from torchvision.transforms import (
    RandomResizedCrop,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    Normalize,
)


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
        # Change output size of model to 10 classes
        model.fc = torch.nn.Linear(640, 10)
        train_dataloader = get_dataloader(
            "cifar10",
            True,
            batch_size=1,
            size=10,
            shuffle=True,
            transforms=self.transfroms(True),
        )
        eval_dataloader = get_dataloader(
            "cifar10", False, batch_size=1, size=5, transforms=self.transfroms()
        )
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

    def transfroms(self, train: bool = False):
        """Load transforms depending on training or evaluation dataset."""
        transforms = []
        if train:
            transforms.append(RandomResizedCrop((32, 32), scale=(0.5, 1), ratio=(1, 1)))
            transforms.append(RandomHorizontalFlip())
            transforms.append(RandomVerticalFlip())
        transforms.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return transforms


if __name__ == "__main__":
    experiment = LpFtExperiment("lp_ft")
    experiment.run()
