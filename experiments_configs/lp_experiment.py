import argparse
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


class LpExperiment(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        batch_size: int = 128,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        super().__init__(experiment_name)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def get_model(self):
        """Get model."""
        return load_model(
            model_name="Addepalli2022Efficient_WRN_34_10",
            dataset="cifar100",
            threat_model="Linf",
        )

    def run(self, device: torch.device = torch.device("cpu")):
        """Run experiment."""
        model = self.get_model()
        # Change output size of model to 10 classes
        model.fc = torch.nn.Linear(640, 10)
        train_dataloader = get_dataloader(
            "cifar10",
            True,
            batch_size=self.batch_size,
            shuffle=True,
            transforms=self.transfroms(True),
        )
        eval_dataloader = get_dataloader(
            "cifar10", False, batch_size=self.batch_size, transforms=self.transfroms()
        )
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        trainer = Trainer(
            model,
            train_dataloader,
            eval_dataloader,
            loss,
            self.epochs,
            optimizer,
            self.experiment_name,
            freeze=self.freeze(),
            device=device,
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

    def freeze(self):
        """Define freeze dictionary."""
        epochs = [i for i in range(self.epochs)]
        return {
            "conv1": epochs,
            "block1": epochs,
            "block2": epochs,
            "block3": epochs,
            "bn1": epochs,
            "relu": epochs,
        }


def main():
    """Command line tool to run experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default=5)
    parser.add_argument("-eps", "--epochs", default=10)
    parser.add_argument("-lr", "--learning_rate", default=0.001)
    parser.add_argument("-device", "--device", default="cpu")

    args = parser.parse_args()
    experiment_name = (
        f"lp_bs_{args.batch_size}_eps_{args.epochs}_lr_{args.learning_rate}"
    )
    experiment = LpExperiment(
        experiment_name, args.batch_size, args.epochs, args.learning_rate
    )
    experiment.run(torch.device(args.device))


if __name__ == "__main__":
    main()
