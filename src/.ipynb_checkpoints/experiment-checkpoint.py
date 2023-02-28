"""Abstract class for experiments."""

from abc import ABC, abstractmethod


class Experiment(ABC):
    """Abstract class for experiments."""

    def __init__(self, experiment_name) -> None:
        self.experiment_name = experiment_name

    @abstractmethod
    def get_model(self):
        """Return pytorch model."""
        pass

    @abstractmethod
    def run(self):
        """Runs experiment."""
        pass
