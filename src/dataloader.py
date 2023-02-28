import os
from urllib.request import urlretrieve
import subprocess
from math import ceil
from pathlib import Path
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor
import torch
from sklearn import preprocessing as p
from typing import List
from src.utils import download_url, unzip, copy_all_files
import glob
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

DATA_DIR = "./data"

import abc
import logging
import os
import subprocess
import torch
import numpy as np
import sklearn.preprocessing
import torchvision

class DisentangledDataset(torch.utils.data.Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.
    Parameters
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self,
                 root,
                 transforms_list=[],
                 logger=logging.getLogger(__name__),
                 subset=1):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = torchvision.transforms.Compose(transforms_list)
        self.logger = logger
        self.subset = subset
        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass

class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].
    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.
    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2
    Parameters
    ----------
    root : string
        Root directory of dataset.
    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.
    """
    urls = {
        "train":
        "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
    }
    files = {"train": "dsprite_train.npz"}
    lat_names = ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([1, 3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = 0
    lat_values = {
        'posX':
        np.array([
            0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032,
            0.19354839, 0.22580645, 0.25806452, 0.29032258, 0.32258065,
            0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
            0.51612903, 0.5483871, 0.58064516, 0.61290323, 0.64516129,
            0.67741935, 0.70967742, 0.74193548, 0.77419355, 0.80645161,
            0.83870968, 0.87096774, 0.90322581, 0.93548387, 0.96774194, 1.
        ]),
        'posY':
        np.array([
            0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032,
            0.19354839, 0.22580645, 0.25806452, 0.29032258, 0.32258065,
            0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
            0.51612903, 0.5483871, 0.58064516, 0.61290323, 0.64516129,
            0.67741935, 0.70967742, 0.74193548, 0.77419355, 0.80645161,
            0.83870968, 0.87096774, 0.90322581, 0.93548387, 0.96774194, 1.
        ]),
        'scale':
        np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
        'orientation':
        np.array([
            0., 0.16110732, 0.32221463, 0.48332195, 0.64442926, 0.80553658,
            0.96664389, 1.12775121, 1.28885852, 1.44996584, 1.61107316,
            1.77218047, 1.93328779, 2.0943951, 2.25550242, 2.41660973,
            2.57771705, 2.73882436, 2.89993168, 3.061039, 3.22214631,
            3.38325363, 3.54436094, 3.70546826, 3.86657557, 4.02768289,
            4.1887902, 4.34989752, 4.51100484, 4.67211215, 4.83321947,
            4.99432678, 5.1554341, 5.31654141, 5.47764873, 5.63875604,
            5.79986336, 5.96097068, 6.12207799, 6.28318531
        ]),
        'shape':
        np.array([1., 2., 3.]),
        'color':
        np.array([1.])
    }

    def __init__(self, root='data/dsprites/', factors_to_use=['shape', 'scale', 'orientation', 'posX', 'posY'], **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)
        if not os.path.exists(f"{root}dsprite_train.npz"): self.download()
        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']
        self.lat_values = sklearn.preprocessing.minmax_scale(self.lat_values)
        self.factors_to_use = factors_to_use
        indices = []
        for x in self.factors_to_use:
            indices.append(np.where(np.array(self.lat_names)==x)[0][0])
        self.lat_values = self.lat_values[:, indices]
        self.lat_sizes = self.lat_sizes[indices]
        self.lat_names = factors_to_use
        """
        for i in range(self.lat_values.shape[1]):
            print(self.lat_names[i], np.unique(self.lat_values[:,i]))
        sys.exit()
        """

        # if self.subset < 1:
        #     n_samples = int(len(self.imgs) * self.subset)
        #     subset = np.random.choice(len(self.imgs), n_samples, replace=False)
        #     self.imgs = self.imgs[subset]
        #     self.lat_values = self.lat_values[subset]

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        subprocess.check_call([
            "curl", "-L",
            type(self).urls["train"], "--output", self.train_data
        ])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        lat_value : np.array
            Array of length len(self.factors_to_use), that gives the value of each factor of variation that is included in self.factors_to_use.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        X = self.transforms(sample)
        y = self.lat_values[idx]
        
        X = torch.tensor(X)
        y = torch.tensor(y)
        return X, y.squeeze(), idx

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

        min_max_scaler = p.MinMaxScaler()
        self.y = min_max_scaler.fit_transform(self.y.reshape(-1, 1)).reshape(-1)
        
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

def get_dataset(dataset_name: str, factors_to_use=['orientation'], train: bool = False, size: int = None):
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
        dataset = DSprites(factors_to_use=factors_to_use)
    if size:
        indices = torch.arange(size)
        dataset = Subset(dataset, indices)
    return dataset


def get_dataloader(
    factors_to_use,
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
    dataset = get_dataset(dataset_name, factors_to_use, train, size)
    transforms = [ToTensor()] + transforms
    if isinstance(dataset, torch.utils.data.Subset):
        dataset.dataset.transform = Compose(transforms)
    else:
        dataset.transform = Compose(transforms)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
