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
from src.dataloader import *
from src.models import *
import foolbox as fb

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

def train_classifier(model, dataset, train_sampler, test_sampler, adv_train,
                     num_epochs=10, fraction_of_labels=1.0, batch_size=1024, 
                     freeze_features=True, subset_seed=None, use_cuda=True, 
                     progress_bar=True, verbose=False, epsilon=[8 / 255],):
   
    model.to(args.device)
    
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
    # Loss and attack
    loss_fn = nn.MSELoss(reduction='mean')
    l_inf_pgd = fb.attacks.LinfPGD(steps=20)
    
    if adv_train:
        print("-------Training a Robust Model-------")
    else:
        print("-------Training a Clean Model-------")
    
    # Train classifier on training set
    model.train()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    
    loss_train = []
    for _ in tqdm(range(num_epochs), disable=not(progress_bar)):
        total_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            X, y, _ = iter_data # ignore indices
            X, y = X.to(args.device), y.to(args.device)
            X = X.resize(X.size(0), 1, 64, 64)
            if adv_train:
                criterion = fb.criteria.Regression(y)
                _, X_pert, success = l_inf_pgd(fmodel, X, criterion=criterion, epsilons=epsilon)
                X_pert[0] = X_pert[0].resize(X.size(0), 1, 64, 64)
                X = X_pert[0]
                
            optimizer.zero_grad()

            predicted_y_logits = model(X)
            loss = loss_fn(predicted_y_logits.squeeze(), y.to(torch.float))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_total += 1
        print(total_loss / num_total)
        print()
        loss_train.append(total_loss / num_total)
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
            criterion = fb.criteria.Regression(y)
            _, X_pert, success = l_inf_pgd(fmodel, X, criterion=criterion, epsilons=epsilon)
            X_pert[0] = X_pert[0].resize(X.size(0), 1, 64, 64)
            
            with torch.no_grad():
                predicted_y_pert_logits = model(X_pert[0])
                predicted_y_logits = model(X)
            
            with torch.no_grad():
                loss_pert = loss_fn(predicted_y_pert_logits.squeeze(), y.to(torch.float))
                loss = loss_fn(predicted_y_logits.squeeze(), y.to(torch.float))
                loss_pert_total += loss_pert.item()
                loss_total += loss.item()
            num_total += 1
    print('perturbed val_loss= ', loss_pert_total/num_total)
    print('benign val_loss= ', loss_total/num_total)
    return model, loss_train

def main(args):
    """Command line tool to run experiment and evaluation."""

    # Initialize a torch dataset, specifying the target latent dimension for
    dSprites_torchdataset = DSprites(
      factors_to_use=args.target_latent
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
    
    model, _ = train_classifier(
        model=supervised_model,
        dataset=dSprites_torchdataset,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
        adv_train=args.adv_train,
        freeze_features=False,
        num_epochs=args.num_epochs,
        )
    if args.adv_train:
        torch.save(model.state_dict(), args.save_dir + f'/{args.target_latent}-robust.pth')
    else:
        torch.save(model.state_dict(), args.save_dir + f'/{args.target_latent}-clean.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="models/dsprites/Linf")
    parser.add_argument('--target_latent', type=str, default="orientation")
    parser.add_argument('--num_epochs', type=int, default=10, help="# of epochs")
    parser.add_argument('--adv_train', type=eval, default=True, help="to train a robust model or not")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    args.target_latent = args.target_latent.split(', ')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main(args)
