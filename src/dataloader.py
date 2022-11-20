import subprocess
from math import ceil
from pathlib import Path
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor
import torch
from typing import List
from src.utils import download_url, unzip, copy_all_files
import glob
from torchvision.io import read_image
from torch.utils.data import Dataset

DATA_DIR = "./data"


class WeatherDataset(Dataset):
    """Dataset of cloudy, rain, shine and sunrise weather images."""
    category2id = {"cloudy": 0, "rain": 1, "shine": 2, "sunrise": 3}
    """Mapping from category name to label id."""
    num_img_per_category = {"cloudy": 300, "rain": 215, "shine": 253, "sunrise": 357}
    """Total number of images per category."""
    split = 0.8
    """Split ratio between train and test. Split is for train set."""
    dataset_folder = Path(DATA_DIR) / "weather_data"
    """Folder to store images in."""

    def __init__(self, train: bool = False, transform: List = None):
        """Initilize dataset.

        :param train: If True train set is return, else test set
        :param transform: Transfroms to apply to images
        """
        self.download()
        self.transform = transform
        self.img_list = []
        for cat, num_imgs in self.num_img_per_category.items():
            split = ceil(num_imgs * self.split)
            start, stop = (0, split) if train else (split, num_imgs)
            for i in range(start, stop):
                self.img_list.append(f"{cat}{i}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.dataset_folder / f"{self.img_list[idx]}.jpg"
        img = read_image(str(img_path)).numpy()
        img_category = "".join([i for i in img_path.stem if not i.isdigit()])
        label = self.category2id[img_category]
        if self.transform:
            img = self.transform(img)
        return img, label

    @classmethod
    def download(cls):
        """Download weather dataset."""
        url = r"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4drtyfjtfy-1.zip"
        image_count = len(glob.glob1(cls.dataset_folder, "*.jpg"))
        if image_count == 1122:
            print("Dataset already downloaded.")
            return
        cls.dataset_folder.mkdir(parents=True, exist_ok=True)
        zip_file = Path(DATA_DIR) / "weather_data.zip"
        download_url(url=url, save_path=zip_file)
        unzip(zip_file, Path(DATA_DIR))
        unzip(Path(DATA_DIR) / "dataset2.zip", Path(DATA_DIR))
        output_folder = Path(DATA_DIR) / "dataset2"
        copy_all_files(output_folder, cls.dataset_folder)


class IntelImageDataset(Dataset):
    """Intel image dataset."""
    category2id = {"buildings": 0, "forest": 1, "glacier": 2, "mountain": 3, "sea": 4, "street": 5}
    """Mapping from category name to label id."""
    dataset_folder = Path(DATA_DIR) / "intel-image-classification"
    """Folder to store images in."""

    def __init__(self, train: bool = False, transform: List = None):
        """Initilize dataset.

        :param train: If True train set is return, else test set
        :param transform: Transfroms to apply to images
        """
        self.download()
        self.transform = transform
        if train:
            path_imgs = self.dataset_folder / "seg_train" / "seg_train"
        else:
            path_imgs = self.dataset_folder / "seg_test" / "seg_test"
        self.img_list = sorted(list(path_imgs.glob('**/*.jpg')))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = read_image(str(img_path)).numpy()
        img_category = img_path.parent.name
        label = self.category2id[img_category]
        if self.transform:
            img = self.transform(img)
        return img, label

    @classmethod
    def download(cls):
        """Download dataset."""
        if cls.dataset_folder.exists():
            print("Dataset already downloaded.")
            return
        subprocess.run(["kaggle", 
                        "datasets", 
                        "download", 
                        "-d", 
                        "puneet6060/intel-image-classification", 
                        "-p", 
                        str(Path(DATA_DIR))])
        unzip(Path(DATA_DIR) / "intel-image-classification.zip", Path(DATA_DIR))


def get_dataset(dataset_name: str, train: bool = False, size: int = None):
    """Load pytorch dataset.

    :param name: Name of dataset
    :param train: If True loads train set, else test set, default False
    :param size: Takes only first size number of items of dataset
    """
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=DATA_DIR, train=train, download=True)
    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=DATA_DIR, train=train, download=True)
    if dataset_name == "weather":
        dataset = WeatherDataset(train=train)
    if dataset_name == "intel_image":
        dataset = IntelImageDataset(train=train)
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
    transforms = [ToTensor()] + transforms
    if isinstance(dataset, torch.utils.data.Subset):
        dataset.dataset.transform = Compose(transforms)
    else:
        dataset.transform = Compose(transforms)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
