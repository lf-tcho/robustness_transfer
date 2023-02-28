import os
import warnings
import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from urllib.request import urlretrieve
import torch
from torch import nn
import torchvision
from src.dataloader import *
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

def train_classifier(model, dataset, train_sampler, test_sampler, 
                     num_epochs=10, fraction_of_labels=1.0, batch_size=1000, 
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
    loss_fn = nn.MSELoss(reduction='mean')
    # Train classifier on training set
    model.train()

    loss_arr = []
    for _ in tqdm(range(num_epochs), disable=not(progress_bar)):
        total_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            X, y, _ = iter_data # ignore indices
            X = X.resize(X.size(0), 1, 64, 64)
           
            optimizer.zero_grad()

            predicted_y_logits = model(X.to(device))
            loss = loss_fn(predicted_y_logits.squeeze(), y.to(device).to(torch.float))
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
            X, y, _ = iter_data # ignore indices
            X = X.resize(X.size(0), 1, 64, 64)
            with torch.no_grad():
                predicted_y_logits = model(X.to(device))
            
            with torch.no_grad():
                loss = loss_fn(predicted_y_logits.squeeze(), y.to(device).to(torch.float))
                loss_total += loss.item()
            num_total += 1
    print('val_loss = ', loss_total/num_total)
    return model, loss_arr

def load_model(ckpt_path, model):
    """Load latest model and get epoch."""
    model.load_state_dict(torch.load(ckpt_path+'.pth'))
    return model

def main(args):
    """Command line tool to run experiment and evaluation."""

    # Initialize a torch dataset, specifying the target latent dimension for
    # the classifier
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
    supervised_model = load_model(f"{args.ckpt_dir}/{args.target_latent}", supervised_model)
    
    if args.train_only_fclayer:
        for p in supervised_model.parameters(): p.requires_grad = False
        supervised_model.fc.weight.requires_grad = True
        supervised_model.fc.bias.requires_grad = True
    for p in supervised_model.parameters():
        print(p.requires_grad)
 
    model, _ = train_classifier(
        model=supervised_model,
        dataset=dSprites_torchdataset,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
        freeze_features=False,
        num_epochs=args.num_epochs,
        )
    if args.train_only_fclayer:
        torch.save(model.state_dict(), f"{args.experiment_folder}/models/{args.target_latent}_lp.pth")
    else:
        torch.save(model.state_dict(), f"{args.experiment_folder}/models/{args.target_latent}_ft.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="bs_128_ds_dsprites_eps_10_lr_0.001_lrs_cosine_tf_method_lp")
    parser.add_argument('--ckpt_dir', type=str, default="models/dsprites/Linf")
    parser.add_argument('--train_only_fclayer', type=eval, default=True)
    parser.add_argument('--target_latent', type=str, default="orientation")
    parser.add_argument('--num_epochs', type=int, default=10,
            help="# of epochs")
    args = parser.parse_args()
    args.target_latent = args.target_latent.split(', ')
    args.experiment_folder = Path("./experiments") / args.exp_name
    
    main(args)
