from dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Trainer class to train a model."""

    def __init__(self, model, dataloader, loss, epochs, optimizer: str = "sgd"):
        self.model = model
        self.dataloader = dataloader
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer

    def get_optimizer(self):
        """Get optimizer."""
        if self.optimizer == "sgd":
            return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        """Train network."""
        optimizer = self.get_optimizer()
        for epoch in range(self.epochs):
            print(epoch)
            for batch in tqdm(self.dataloader):
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                optimizer.step()
                # TODO: Add tensorboard progress tracking


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
