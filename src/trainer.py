"""Training functionality."""


import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

CKPT_NAME = "ckpt"


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
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.model = model.to(self.device)
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
        latest_epoch = self.load_model()
        iteration = (latest_epoch + 1) * len(self.train_dataloader)
        for epoch in range(latest_epoch + 1, self.epochs):
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
            self.save_model(f"{CKPT_NAME}_{epoch}")

    def step(self, inputs, labels):
        """Do one batch step."""
        metrics = {}
        model_output = self.model(inputs.to(self.device))
        metrics["loss"] = self.loss(model_output, labels.to(self.device))
        return metrics

    def save_model(self, name):
        """Save model.

        :param name: Name of checkpoint without file extension
        """
        torch.save(self.model.state_dict(), self.experiment_folder / f"{name}.pth")

    def load_model(self):
        """Load latest model and get epoch."""
        ckpts = sorted(list(self.experiment_folder.glob("*.pth")))
        if ckpts:
            latest_epoch = int(ckpts[-1].stem.split("_")[-1])
            self.model.load_state_dict(
                torch.load(self.experiment_folder / f"{CKPT_NAME}_{latest_epoch}.pth")
            )
            print(f"Model checkpoint {CKPT_NAME}_{latest_epoch}.pth loaded.")
            return latest_epoch
        else:
            return -1
