import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms


DATA_DIR = "./data"


def get_dataset(dataset_name: str, train: bool = False):
    """Load pytorch dataset.

    :param name: Name of dataset
    :param train: If True loads train set, else test set, default False
    """
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )


def get_dataloader(dataset_name: str, train: bool = False, batch_size: int = 1):
    """Get dataloader for given dataset name.

    :param name: Name of dataset
    :param train: If True loads train set, else test set, default False
    :param batch_size: Batch size of dataloader
    """
    dataset = get_dataset(dataset_name, train)
    return DataLoader(dataset, batch_size)
