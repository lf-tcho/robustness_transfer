import argparse
from src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from src.trainer import Trainer
from src.experiment import Experiment
import torch
from torchvision.transforms import Normalize


class LpExperiment(Experiment):
    """Experiment for linear probing."""

    def __init__(
        self,
        experiment_name,
        batch_size: int = 128,
        epochs: int = 10,
        learning_rate: float = 0.001,
        tf_method: str = "lp",
        lp_epochs: int = 0,
        lr_scheduler: str = None,
    ):
        """Initilize LpExperiment.

        :param experiment_name: Name of experiment
        :param batch_size: Batch size for training
        :param epochs: Training epochs
        :param learning_rate: Learning rate for training
        :param tf_method: Transfer learning method ("lp", "lp_ft", None)
        :param lp_epochs: Number of epochs of linear probing before full fine-tuning. Only used for method lp_ft
        :param lr_scheduler: Learning rate scheduler used for training
        """
        super().__init__(experiment_name)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tf_method = tf_method
        self.lp_epochs = lp_epochs
        self.lr_scheduler = lr_scheduler

    def get_model(self):
        """Get model."""
        model = load_model(
            model_name="Addepalli2022Efficient_WRN_34_10",
            dataset="cifar100",
            threat_model="Linf",
        )
        # Change output size of model to 10 classes
        model.fc = torch.nn.Linear(640, 10)
        return model

    def get_lr_scheduler(self, optimizer):
        """Get learning rate scheduler based of name for lr scheduler.

        :param optimizer: Optimizer used for training.
        """
        if self.lr_scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.epochs)
        else:
            return

    def run(self, device: torch.device = torch.device("cpu")):
        """Run experiment."""
        model = self.get_model()
        train_dataloader = get_dataloader(
            "cifar10",
            True,
            batch_size=self.batch_size,
            shuffle=True,
            transforms=self.transforms(True),
        )
        eval_dataloader = get_dataloader(
            "cifar10", False, batch_size=self.batch_size, transforms=self.transforms()
        )
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        trainer = Trainer(
            model,
            train_dataloader,
            eval_dataloader,
            nn.CrossEntropyLoss(),
            self.epochs,
            optimizer,
            self.experiment_name,
            freeze=self.freeze(),
            device=device,
            lr_scheduler=self.get_lr_scheduler(optimizer)
        )
        trainer.train()

    def transforms(self, train: bool = False):
        """Load transforms depending on training or evaluation dataset."""
        return [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def freeze(self):
        """Define freeze dictionary."""
        if self.tf_method == "lp":
            epochs = [i for i in range(self.epochs)]
            return {
                "conv1": epochs,
                "block1": epochs,
                "block2": epochs,
                "block3": epochs,
                "bn1": epochs,
                "relu": epochs,
            }
        elif self.tf_method == "lp_ft":
            epochs = [i for i in range(self.lp_epochs)]
            return {
                "conv1": epochs,
                "block1": epochs,
                "block2": epochs,
                "block3": epochs,
                "bn1": epochs,
                "relu": epochs,
            }
        else:
            return None


def main():
    """Command line tool to run experiment and evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default=5, type=int)
    parser.add_argument("-eps", "--epochs", default=10, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    parser.add_argument("-device", "--device", default="cpu")
    parser.add_argument("-method", "--tf_method", default="lp", type=str)
    parser.add_argument("-eval", "--eval", default=0, type=int)
    parser.add_argument("-train", "--train", default=0, type=int)
    parser.add_argument("-evaleps", "--evaleps", default=None, type=int)
    parser.add_argument("-evalbs", "--evalbs", default=32, type=int)
    parser.add_argument("-evaldssize", "--evaldssize", default=None, type=int)
    parser.add_argument("-lp_epochs", "--lp_epochs", default=0, type=int)
    parser.add_argument("-lr_scheduler", "--lr_scheduler", default=None, type=str)
    args = parser.parse_args()
    experiment_name = f"bs_{args.batch_size}_eps_{args.epochs}_lr_{args.learning_rate}_tf_method_{args.tf_method}_lrs_{args.lr_scheduler}"
    if args.tf_method == "lp_ft":
        experiment_name += f"_lpeps_{args.lp_epochs}"
    experiment = LpExperiment(
        experiment_name,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.tf_method,
        args.lp_epochs,
    )

    if args.train:
        experiment.run(torch.device(args.device))

    if args.eval:
        from src.evaluator import Evaluator

        transforms = [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader = get_dataloader(
            "cifar10",
            False,
            args.evalbs,
            args.evaldssize,
            transforms,
        )
        evaluator = Evaluator(
            experiment,
            dataloader,
            epoch=args.evaleps,
            device=torch.device(args.device),
        )
        evaluator.eval()


if __name__ == "__main__":
    main()
