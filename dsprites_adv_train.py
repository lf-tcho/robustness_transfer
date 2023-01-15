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

import os
import argparse
import copy
from tqdm.auto import tqdm

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
                loss = F.mse_loss(logits, y.resize(1, y.size(0)).float())
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


def train(net, train_loader, optimizer, adversary, criterion, epoch, args):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    for iter_data in enumerate(train_loader):
        inputs, targets, _ = iter_data[1] # ignore indices
        inputs = inputs.resize(inputs.size(0), 1, 64, 64)
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets.resize(1, targets.size(0)).float())
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        total += targets.size(0)
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train loss:', loss.item())
        batch_idx += 1 
    print('Total adversarial train loss:', train_loss)

def test(net, test_loader, optimizer, adversary, criterion, epoch, args):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    batch_idx = 0
    with torch.no_grad():
        for iter_data in enumerate(test_loader):
            inputs, targets, _ = iter_data[1] # ignore indices
            inputs = inputs.resize(inputs.size(0), 1, 64, 64)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets.resize(1, targets.size(0)).float())
            benign_loss += loss.item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test loss:', loss.item())

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets.resize(1, targets.size(0)).float())
            adv_loss += loss.item()

            if batch_idx % 10 == 0:
                print('Current adversarial test loss:', loss.item())
            batch_idx += 1 
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {
        'net': net.state_dict()
    }
    torch.save(state, args.save_dir + f'/{args.target_latent}.pth')
    print('Model Saved!')

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

    # Define datasets and dataloaders
    train_loader = torch.utils.data.DataLoader(
        dSprites_torchdataset, batch_size=args.batch_size, sampler=train_sampler,
        )
    test_loader = torch.utils.data.DataLoader(
        dSprites_torchdataset, batch_size=args.batch_size, sampler=test_sampler,
        )

    print(f"Dataset size: {len(train_sampler)} training, "
          f"{len(test_sampler)} test images")

    net = Model_dsprites()
    net = net.to(args.device)
    net = torch.nn.DataParallel(net)

    adversary = LinfPGDAttack(net, args.alpha, args.epsilon, args.k)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0002)


    for epoch in tqdm(range(0, args.num_epochs)):
        adjust_learning_rate(optimizer, epoch, args.learning_rate)
        train(net, train_loader, optimizer, adversary, criterion, epoch, args)
        test(net, test_loader, optimizer, adversary, criterion, epoch, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="models/dsprites/Linf/")
    parser.add_argument('--target_latent', type=str, default="scale")
    parser.add_argument('--num_epochs', type=int, default=200,)
    parser.add_argument('--learning_rate', type=float, default=1e-2,)
    parser.add_argument('--epsilon', type=float, default=0.0314,)
    parser.add_argument('--k', type=int, default=7,)
    parser.add_argument('--alpha', type=float, default=0.00784,)
    parser.add_argument('--batch_size', type=int, default=128,)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    main(args)
