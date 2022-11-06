import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor
import torch
from typing import List


DATA_DIR = "./data"


def get_dataset(dataset_name: str, train: bool = False, size: int = None):
    """Load pytorch dataset.

    :param name: Name of dataset
    :param train: If True loads train set, else test set, default False
    :param size: Takes only first size number of items of dataset
    """
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=Compose([ToTensor()]),
        )
    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=Compose([ToTensor()]),
        )
    if size:
        indices = torch.arange(size)
        dataset = Subset(dataset, indices)
    return dataset


def get_dataloader(
    dataset_name: str,
    train: bool = False,
    batch_size: int = 1,
    size: int = None,
    transforms: List = [],
    shuffle: bool = False,
):
    """Get dataloader for given dataset name.

    :param name: Name of dataset
    :param train: If True loads train set, else test set, default False
    :param batch_size: Batch size of dataloader
    :param size: Takes only first size number of items of dataset
    """
    dataset = get_dataset(dataset_name, train, size)
    if transforms:
        dataset.transform = Compose(transforms)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
