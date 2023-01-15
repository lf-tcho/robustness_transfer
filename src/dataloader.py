import os
from urllib.request import urlretrieve
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
from PIL import Image
import numpy as np

DATA_DIR = "./data"

class dSpritesTorchDataset(torch.utils.data.Dataset):
    def __init__(self, path='./data/dsprites/', train=True, target_latent="shape"):
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        
        self.path = path    
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            urlretrieve(url, os.path.join(path, 'dsprites.npz'))
        self.npz = self.load_data()
        print('done data loading!')

        metadata = self.npz["metadata"][()]
        self.latent_class_values = metadata["latents_possible_values"]
        self.latent_class_names = metadata["latents_names"]
        self.latent_classes = self.npz["latents_classes"][()]
        
        self.target_latent = target_latent

        self.X = self.images()
        self.y = self.get_latent_classes(
            latent_class_names=target_latent
            ).squeeze()

        if train:
            self.X = self.X[:int(len(self.X)*0.8)]
            self.y = self.y[:int(len(self.y)*0.8)] 
        else:
            self.X = self.X[int(len(self.X)*0.8):]
            self.y = self.y[int(len(self.y)*0.8):] 

        self.num_classes = \
            len(self.latent_class_values[self.target_latent])

        self.num_samples = len(self.X)

    def load_data(self):
        dataset_zip = np.load(os.path.join(self.path, 'dsprites.npz'),
                    encoding="latin1", allow_pickle=True)
        return dataset_zip
  
    def images(self):
        """
        Lazily load and returns all dataset images.
        - self._images: (3D np array): images (image x height x width)
        """

        if not hasattr(self, "_images"):
            self._images = self.npz["imgs"][()]
        return self._images

    def _check_class_name(self, latent_class_name="shape"):
        """
        self._check_class_name()
        Raises an error if latent_class_name is not recognized.
        Optional args:
        - latent_class_name (str): name of latent class to check. 
            (default: "shape")
        """
        if latent_class_name not in self.latent_class_names:
            latent_names_str = ", ".join(self.latent_class_names)
            raise ValueError(
                f"{latent_class_name} not recognized as a latent class name. "
                f"Must be in: {latent_names_str}."
                )


    def get_latent_name_idxs(self, latent_class_names=None):
        """
        self.get_latent_name_idxs()
        Returns indices for latent class names.
        Optional args:
        - latent_class_names (str or list): name(s) of latent class(es) for 
            which to return indices. Order is preserved. If None, indices 
            for all latents are returned. (default: None)
        
        Returns:
        - (list): list of latent class indices
        """

        if latent_class_names is None:
            return np.arange(len(self.latent_class_names))

        if not isinstance(latent_class_names, (list, tuple)):
            latent_class_names = [latent_class_names]       
        
        latent_name_idxs = []
        for latent_class_name in latent_class_names:
            self._check_class_name(latent_class_name)
            latent_name_idxs.append(
                self.latent_class_names.index(latent_class_name)
                ) 

        return latent_name_idxs  


    def get_latent_classes(self, indices=None, latent_class_names=None):
        """
        self.get_latent_classes()
        Returns latent classes for each image.
        Optional args:
        - indices (array-like): image indices for which to return latent 
            class values. Order is preserved. If None, all are returned 
            (default: None).
        - latent_class_names (str or list): name(s) of latent class(es) 
            for which to return latent class values. Order is preserved. 
            If None, values for all latents are returned. (default: None)
        
        Returns:
        - (2D np array): array of latent classes (img x latent class)
        """

        if indices is not None:
            indices = np.asarray(indices)
        else:
            indices = slice(None)

        latent_class_name_idxs = self.get_latent_name_idxs(latent_class_names)

        return self.latent_classes[indices][:, latent_class_name_idxs]


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        y = self.y[idx]

        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, y, idx

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
            start, stop = (1, split) if train else (split, num_imgs)
            for i in range(start, stop):
                self.img_list.append(f"{cat}{i}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.dataset_folder / f"{self.img_list[idx]}.jpg"
        if not img_path.exists():
            img_path = img_path.with_suffix('.jpeg')
        img = np.array(np.asarray(Image.open(str(img_path)).convert("RGB")))
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
        img = read_image(str(img_path)).permute(1,2,0).numpy()
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
        unzip(Path(DATA_DIR) / "intel-image-classification.zip", cls.dataset_folder)

class Fashion(datasets.FashionMNIST):

    def __getitem__(self, index: int):
        """
        :param index: Index
        :return: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L").convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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
    if dataset_name == "fashion":
        dataset = Fashion(root=DATA_DIR, train=train, download=True)
    if dataset_name == "weather":
        dataset = WeatherDataset(train=train)
    if dataset_name == "intel_image":
        dataset = IntelImageDataset(train=train)
    if dataset_name=="dsprites":
        dataset = dSpritesTorchDataset(train=train)
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
