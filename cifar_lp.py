import os
import warnings
import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from urllib.request import urlretrieve
import torch
from torch import nn
from torchvision.transforms import Normalize, Resize
from torchvision.transforms import Compose, ToTensor
import torchvision.datasets as datasets
from src.dataloader import *
from src.models import Model_dsprites
DATA_DIR = './data'

def train_classifier(model, train_dataset, test_dataset, 
                     num_epochs=10,  batch_size=1024, 
                     freeze_features=True, subset_seed=None, use_cuda=True, 
                     progress_bar=True, verbose=False):
   
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    model.to(device)
    
    # Define datasets and dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        )

    # Define loss and optimizers
    train_parameters = model.parameters()

    optimizer = torch.optim.Adam(train_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
        )
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # Train classifier on training set
    model.train()

    loss_arr = []
    for _ in tqdm(range(num_epochs), disable=not(progress_bar)):
        total_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            X, y = iter_data # ignore indices
            # X = X.resize(X.size(0), 1, 64, 64)
           
            optimizer.zero_grad()

            predicted_y_logits = model(X.to(device))
            loss = loss_fn(predicted_y_logits.squeeze(), y.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_total += 1

        loss_arr.append(total_loss / num_total)
        scheduler.step()
    # Calculate prediction accuracy on training and test sets
    model.eval()

    accuracies = []
    for _, dataloader in enumerate((train_dataloader, test_dataloader)):
        num_correct = 0
        num_total = 0
        loss_total = 0
        for iter_data in dataloader:
            X, y = iter_data # ignore indices
            # X = X.resize(X.size(0), 1, 64, 64)
            with torch.no_grad():
                predicted_y_logits = model(X.to(device))
            
            with torch.no_grad():
                loss = loss_fn(predicted_y_logits.squeeze(), y.to(device))
                loss_total += loss.item()
            num_total += 1
    print('val_loss = ', loss_total/num_total)
    return model, loss_arr

def load_model(model_name="Addepalli2022Efficient_WRN_34_10"):
    """Get model"""
    from robustbench.utils import load_model
    model = load_model(
        model_name=model_name,
        dataset="cifar100",
        threat_model="Linf",
    )
    # Change output size of model to 10 classes
    model.fc = torch.nn.Linear(640, 10)
    return model
    
def transforms():
    """Load transforms depending on training or evaluation dataset."""
    return [ToTensor(), Resize((32, 32)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
def main(args):
    """Command line tool to run experiment and evaluation."""

    # CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    
    # transforms
    train_dataset.transform = Compose(transforms())
    test_dataset.transform = Compose(transforms())

    # Initialize a core encoder network on which the classifier will be added
    supervised_model = load_model()
    
    if args.train_only_fclayer:
        for p in supervised_model.parameters(): p.requires_grad = False
        supervised_model.fc.weight.requires_grad = True
        supervised_model.fc.bias.requires_grad = True
    for p in supervised_model.parameters():
        print(p.requires_grad)
 
    model, _ = train_classifier(
        model=supervised_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        freeze_features=False,
        num_epochs=args.num_epochs,
        )
    os.makedirs(f"{args.experiment_folder}/models/", exist_ok=True)
    if args.train_only_fclayer:
        torch.save(model.state_dict(), f"{args.experiment_folder}/models/cifar100-cifar10_lp.pth")
    else:
        torch.save(model.state_dict(), f"{args.experiment_folder}/models/cifar100-cifar10_ft.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="bs_128_ds_cifar100-cifar10_eps_10_lr_0.001_lrs_cosine_tf_method_lp")
    parser.add_argument('--train_only_fclayer', type=eval, default=True)
    parser.add_argument('--num_epochs', type=int, default=10,
            help="# of epochs")
    args = parser.parse_args()
    args.experiment_folder = Path("./experiments") / args.exp_name
    
    main(args)
