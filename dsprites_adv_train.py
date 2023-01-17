<<<<<<< HEAD
=======
'''
Reference : https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/blob/master/pgd_adversarial_training.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

>>>>>>> 80b047704fd1beb162ad2643f57285bc171bdd2d
import os
import warnings
import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from urllib.request import urlretrieve
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from src.dataloader import dSpritesTorchDataset
from src.models import Model_dsprites

def train_test_split_idx(dataset, fraction_train=0.8, randst=None,):

    if 1 <= fraction_train <= 0:
        raise ValueError(
            "fraction_train must be between 0 and 1, inclusively, but "
            f"found {fraction_train}."
            )

    train_size = int(fraction_train * len(dataset))

    if isinstance(randst, int):
        randst = torch.random.manual_seed(randst)

    all_indices = torch.randperm(len(dataset), generator=randst)

    train_indices = all_indices[: train_size]
    test_indices = all_indices[train_size :]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler

class LinfPGDAttack(object):
    def __init__(self, model, alpha, epsilon, k):
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon
        self.k = k

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.mse_loss(logits.squeeze(), y.float())
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

def train_classifier(model, adversary, dataset, train_sampler, test_sampler, 
                     num_epochs=10, fraction_of_labels=1.0, batch_size=1024, 
                     freeze_features=True, subset_seed=None, use_cuda=True, 
                     progress_bar=True, verbose=False):
   
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    model.to(device)
    
    # Define datasets and dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
        )
    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
        )

    # Define loss and optimizers
    train_parameters = model.parameters()

    optimizer = torch.optim.Adam(train_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
        )
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    # Train classifier on training set
    model.train()

    loss_pert_train = []
    for _ in tqdm(range(num_epochs), disable=not(progress_bar)):
        total_pert_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            X, y, _ = iter_data # ignore indices
            X, y = X.to(args.device), y.to(args.device)
            X = X.resize(X.size(0), 1, 64, 64)
            X_pert = adversary.perturb(X, y)
            
            optimizer.zero_grad()

            predicted_y_pert_logits = model(X_pert)
            loss_pert = loss_fn(predicted_y_pert_logits.squeeze(), y.to(torch.float))
            loss_pert.backward()
            optimizer.step()

            total_pert_loss += loss_pert.item()
            num_total += y.size(0)

        loss_pert_train.append(total_pert_loss / num_total)
        scheduler.step()
    # Calculate prediction accuracy on training and test sets
    model.eval()

    accuracies = []
    for _, dataloader in enumerate((train_dataloader, test_dataloader)):
        num_correct = 0
        num_total = 0
        loss_pert_total = 0
        loss_total = 0
        for iter_data in dataloader:
            X, y, _ = iter_data # ignore indices
            X, y = X.to(args.device), y.to(args.device)
            X = X.resize(X.size(0), 1, 64, 64)
            X_pert = adversary.perturb(X, y)
            
            with torch.no_grad():
                predicted_y_pert_logits = model(X_pert)
                predicted_y_logits = model(X)
            
            with torch.no_grad():
                loss_pert = loss_fn(predicted_y_pert_logits.squeeze(), y.to(torch.float))
                loss = loss_fn(predicted_y_logits.squeeze(), y.to(torch.float))
                loss_pert_total += loss_pert.item()
                loss_total += loss.item()
            num_total += y.size(0)
    print('perturbed val_loss= ', loss_pert_total/num_total)
    print('benign val_loss= ', loss_total/num_total)
    return model, loss_pert_train

def main(args):
    """Command line tool to run experiment and evaluation."""

    # Initialize a torch dataset, specifying the target latent dimension for
    # the classifier
    dSprites_torchdataset = dSpritesTorchDataset(
      target_latent=args.target_latent
      )

    # Initialize a train_sampler and a test_sampler to keep the two sets
    # consistently separate
    train_sampler, test_sampler = train_test_split_idx(
      dSprites_torchdataset,
      fraction_train=0.8,  # 80:20 data split
      )

    print(f"Dataset size: {len(train_sampler)} training, "
          f"{len(test_sampler)} test images")

    # Initialize a core encoder network on which the classifier will be added
    supervised_model = Model_dsprites()
    # Initialize an attack
    adversary = LinfPGDAttack(supervised_model, args.alpha, args.epsilon, args.k)
    
    model, _ = train_classifier(
        model=supervised_model,
        adversary=adversary,
        dataset=dSprites_torchdataset,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
        freeze_features=False,
        num_epochs=args.num_epochs,
        )

    torch.save(model.state_dict(), args.save_dir + f'/{args.target_latent}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="models/dsprites/Linf")
    parser.add_argument('--target_latent', type=str, default="scale")
    parser.add_argument('--num_epochs', type=int, default=30, help="# of epochs")
    parser.add_argument('--epsilon', type=float, default=0.0314,)
    parser.add_argument('--k', type=int, default=7,)
    parser.add_argument('--alpha', type=float, default=0.00784,)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
<<<<<<< HEAD
    
=======
    cudnn.benchmark = True

>>>>>>> 80b047704fd1beb162ad2643f57285bc171bdd2d
    main(args)
