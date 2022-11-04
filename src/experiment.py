"""Abstract class for experiments."""

from abc import ABC, abstractmethod


class Experiment(ABC):
    """Abstract class for experiments."""

    @abstractmethod
    def get_model(self):
        """Return pytorch model."""
        pass

    @abstractmethod
    def run(self):
        """Runs experiment."""
        pass
