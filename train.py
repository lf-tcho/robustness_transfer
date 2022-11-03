from dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

LOG_DIR = "./log"


class Trainer:
    """Trainer class to train a model."""

    def __init__(
        self,
        model,
        dataloader,
        loss,
        epochs,
        experiment_name: str = "test",
        optimizer: str = "sgd",
    ):
        self.model = model
        self.dataloader = dataloader
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.log_dir = Path(LOG_DIR) / experiment_name

    def get_optimizer(self):
        """Get optimizer."""
        if self.optimizer == "sgd":
            return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        """Train network."""
        writer = SummaryWriter(log_dir=self.log_dir)
        optimizer = self.get_optimizer()
        iteration = 0
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            for batch in tqdm(self.dataloader):
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), iteration)
                iteration += self.dataloader.batch_size


def main():
    model = load_model(
        model_name="Carmon2019Unlabeled", dataset="cifar10", threat_model="Linf"
    )
    dataloader = get_dataloader("cifar10", False, batch_size=10)
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(model, dataloader, loss, 10)
    trainer.train()


if __name__ == "__main__":
    main()
