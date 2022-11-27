"""Training functionality."""
import copy

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from typing import List
import operator
import torch.nn as nn

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
        freeze: dict = None,
        lr_scheduler=None,
    ):
        """Initilize Trainer.

        :param model: Model used for training
        :param train_dataloader: Training dataloader
        :param eval_dataloader: Evaluation dataloader
        :param loss: Loss function
        :param epochs: Number of training epochs
        :param optimizer: Optimizer for training
        :param experiment_name: Experiment name used for logging
        :param device: Device used for running Trainer
        :param freeze: Dictionary with model modules to freeze
        :param lr_scheduler: Learning rate scheduler. Note: Optimizer must be
            passed to learning rate scheduler before passing it to Trainer.
        """
        self.device = device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.experiment_folder = Path("./experiments") / experiment_name
        self.freeze = freeze
        self.lr_scheduler = lr_scheduler

    def train(self):
        """Train network."""
        writer = SummaryWriter(log_dir=self.experiment_folder / "logs")
        optimizer = self.optimizer
        latest_epoch = self.load_model()
        iteration = (latest_epoch + 1) * len(self.train_dataloader)
        for epoch in range(latest_epoch + 1, self.epochs):
            if self.freeze:
                self.freeze_model(self.model, epoch, self.freeze)
            # Train one epoch
            self.model.train()
            for inputs, labels in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch} (train): "
            ):
                iteration += self.train_dataloader.batch_size
                optimizer.zero_grad()
                metrics = self.step(inputs.to(self.device), labels.to(self.device))
                loss = metrics["loss"]
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), iteration)
                writer.add_scalar(
                    "Accuracy/train", metrics["accuracy"].item(), iteration
                )
                if self.lr_scheduler:
                    writer.add_scalar(
                        "Parameter/LR",
                        np.array(self.lr_scheduler.get_last_lr()),
                        iteration,
                    )
                    self.lr_scheduler.step()
                else:
                    writer.add_scalar(
                        "Parameter/LR", optimizer.param_groups[0]["lr"], iteration
                    )

            # Evaluate model
            self.model.eval()
            losses = []
            accuracies = []
            with torch.no_grad():
                for inputs, labels in tqdm(
                    self.eval_dataloader, desc=f"Epoch {epoch} (eval): "
                ):
                    metrics = self.step(inputs.to(self.device), labels.to(self.device))
                    losses.append(metrics["loss"].item())
                    accuracies.append(metrics["accuracy"].item())
            loss = np.mean(losses)
            accuracy = np.mean(accuracies)
            writer.add_scalar("Loss/val", loss, iteration)
            writer.add_scalar("Accuracy/val", accuracy, iteration)
            self.save_model(f"{CKPT_NAME}_{epoch}")

    def step(self, inputs, labels):
        """Do one batch step."""
        metrics = {}
        model_output = self.model(inputs.to(self.device))
        metrics["loss"] = self.loss(model_output, labels.to(self.device))
        _, indices = model_output.max(dim=1)
        metrics["accuracy"] = torch.sum(indices == labels) / inputs.shape[0]
        return metrics

    def save_model(self, name):
        """Save model.

        :param name: Name of checkpoint without file extension
        """
        torch.save(self.model.state_dict(), self.experiment_folder / f"{name}.pth")

    def load_model(self):
        """Load latest model and get epoch."""
        ckpts = sorted(
            list(self.experiment_folder.glob("*.pth")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if ckpts:
            latest_epoch = int(ckpts[-1].stem.split("_")[-1])
            ckpt = f"{CKPT_NAME}_{latest_epoch}.pth"
            self.model.load_state_dict(torch.load(self.experiment_folder / ckpt))
            print(f"Model checkpoint {ckpt} loaded.")
            return latest_epoch
        else:
            return -1

    def freeze_model(self, model, epoch: int, freeze: dict = None):
        """Freeze part of model.

        :param model: A pytorch module
        :param epoch: Current epoch
        :param freeze: Keys are module names (separated by dots).
            Values are epochs where the model should be frozen.
        """
        if freeze:
            for module_name, epochs in freeze.items():
                module = operator.attrgetter(module_name)(model)
                if epoch in epochs:
                    module.requires_grad_(False)
                else:
                    module.requires_grad_(True)
        else:
            self.model.requires_grad_(True)


class TrainerLwF(Trainer):
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
        freeze: dict = None,
        lr_scheduler=None,
    ):
        """Initilize Trainer.

        :param model: Model used for training
        :param train_dataloader: Training dataloader
        :param eval_dataloader: Evaluation dataloader
        :param loss: Loss function
        :param epochs: Number of training epochs
        :param optimizer: Optimizer for training
        :param experiment_name: Experiment name used for logging
        :param device: Device used for running Trainer
        :param freeze: Dictionary with model modules to freeze
        :param lr_scheduler: Learning rate scheduler. Note: Optimizer must be
            passed to learning rate scheduler before passing it to Trainer.
        """
        self.device = device
        self.model = model.to(self.device)
        self.teacher_model = copy.deepcopy(model).to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.experiment_folder = Path("./experiments") / experiment_name
        self.freeze = freeze
        self.lr_scheduler = lr_scheduler

    def train(self):
        """Train network."""
        writer = SummaryWriter(log_dir=self.experiment_folder / "logs")
        optimizer = self.optimizer
        latest_epoch = self.load_model()
        iteration = (latest_epoch + 1) * len(self.train_dataloader)
        for epoch in range(latest_epoch + 1, self.epochs):
            if self.freeze:
                self.freeze_model(self.model, epoch, self.freeze)
            # Train one epoch
            self.model.train()
            self.teacher_model.eval()
            for inputs, labels in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch} (train): "
            ):
                iteration += self.train_dataloader.batch_size
                optimizer.zero_grad()
                metrics = self.step(inputs.to(self.device), labels.to(self.device))
                loss = metrics["loss"]
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), iteration)
                writer.add_scalar("Accuracy/train", metrics["accuracy"].item(), iteration)
                writer.add_scalar("cross_entropy/train", metrics["cross_entropy"].item(), iteration)
                writer.add_scalar("feature_difference/train", metrics["feature_difference"].item(), iteration)
                writer.add_scalar(
                    "Accuracy/train", metrics["accuracy"].item(), iteration
                )
                if self.lr_scheduler:
                    writer.add_scalar(
                        "Parameter/LR",
                        np.array(self.lr_scheduler.get_last_lr()),
                        iteration,
                    )
                    self.lr_scheduler.step()
                else:
                    writer.add_scalar(
                        "Parameter/LR", optimizer.param_groups[0]["lr"], iteration
                    )

            # Evaluate model
            self.model.eval()
            losses = []
            accuracies = []
            with torch.no_grad():
                for inputs, labels in tqdm(
                    self.eval_dataloader, desc=f"Epoch {epoch} (eval): "
                ):
                    metrics = self.step(inputs.to(self.device), labels.to(self.device))
                    losses.append(metrics["loss"].item())
                    accuracies.append(metrics["accuracy"].item())
            loss = np.mean(losses)
            accuracy = np.mean(accuracies)
            writer.add_scalar("Loss/val", loss, iteration)
            writer.add_scalar("Accuracy/val", accuracy, iteration)
            self.save_model(f"{CKPT_NAME}_{epoch}")

    def step(self, inputs, labels):
        """Do one batch step."""
        metrics = {}
        model_output = self.model(inputs.to(self.device))
        teacher_output = self.teacher_model(inputs.to(self.device))
        metrics["loss"] = self.loss(
            model_output, teacher_output, labels.to(self.device)
        )
        metrics["loss"] = self.loss(model_output, teacher_output, labels.to(self.device))
        metrics["cross_entropy"] = nn.CrossEntropyLoss()(model_output[0], labels.to(self.device))
        metrics["feature_difference"] = torch.mean(torch.norm(model_output[1]-teacher_output[1], dim=1))
        _, indices = model_output[0].max(dim=1)
        metrics["accuracy"] = torch.sum(indices == labels) / inputs.shape[0]
        return metrics

    def load_model(self):
        """Load latest model and get epoch."""
        ckpts = sorted(list(self.experiment_folder.glob("*.pth")))
        if ckpts:
            latest_epoch = int(ckpts[-1].stem.split("_")[-1])
            ckpt = f"{CKPT_NAME}_{latest_epoch}.pth"
            self.model.load_state_dict(torch.load(self.experiment_folder / ckpt))
            print(f"Model checkpoint {ckpt} loaded.")
            return latest_epoch
        else:
            return -1
