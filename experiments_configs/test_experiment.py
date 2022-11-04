from src.dataloader import get_dataloader
from robustbench.utils import load_model
import torch.optim as optim
import torch.nn as nn
from src.train import Trainer
from src.experiment import Experiment


class TestExperiment(Experiment):
    def get_model(self):
        return load_model(
            model_name="Carmon2019Unlabeled", dataset="cifar10", threat_model="Linf"
        )

    def run(self):
        model = self.get_model()
        train_dataloader = get_dataloader("cifar10", True, batch_size=10, size=100)
        eval_dataloader = get_dataloader("cifar10", False, batch_size=10, size=50)
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
    experiment = TestExperiment("test")
    experiment.run()
