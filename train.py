from dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np


class Trainer:
    """Trainer class to train a model."""

    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        loss,
        epochs,
        optimizer,
        experiment_name: str = "test",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.experiment_folder = Path("./experiments") / experiment_name

    def train(self):
        """Train network."""
        writer = SummaryWriter(log_dir=self.experiment_folder / "logs")
        optimizer = self.optimizer
        iteration = 0
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            # Train one epoch
            for inputs, labels in tqdm(self.train_dataloader):
                iteration += self.train_dataloader.batch_size
                optimizer.zero_grad()
                metrics = self.step(inputs, labels)
                loss = metrics["loss"]
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), iteration)

            # Evaluate model
            losses = []
            with torch.no_grad():
                for inputs, labels in tqdm(self.eval_dataloader):
                    metrics = self.step(inputs, labels)
                    losses.append(metrics["loss"].item())
            loss = np.mean(losses)
            writer.add_scalar("Loss/val", loss, iteration)
        self.save_model(f"ckpt{epoch}")

    def step(self, inputs, labels):
        """Do one batch step."""
        metrics = {}
        model_output = self.model(inputs)
        metrics["loss"] = self.loss(model_output, labels)
        return metrics

    def save_model(self, name):
        """Save model."""
        torch.save(self.model.state_dict(), self.experiment_folder / f"{name}.pth")


def main():
    model = load_model(
        model_name="Carmon2019Unlabeled", dataset="cifar10", threat_model="Linf"
    )
    train_dataloader = get_dataloader("cifar10", True, batch_size=10, size=100)
    eval_dataloader = get_dataloader("cifar10", False, batch_size=10, size=50)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = Trainer(model, train_dataloader, eval_dataloader, loss, 10, optimizer)
    trainer.train()


if __name__ == "__main__":
    main()
