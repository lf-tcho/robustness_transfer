import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
import foolbox as fb

from src.dataloader import get_dataloader
from src.models import *


def get_model(model_name="Addepalli2022Efficient_WRN_34_10"):
    """Get model."""
    from robustbench.utils import load_model
    model = load_model(
        model_name=model_name,
        dataset="cifar100",
        threat_model="Linf",
    )
    # Change output size of model to 10 classes
    model.fc = torch.nn.Linear(640, 10)
    return model

def load_model(model, ckpt_path):
    """Load latest model and get epoch."""
    model.load_state_dict(torch.load(ckpt_path))
    return model

if __name__ == "__main__":
    """
    Visualizes a sample and its adversarial counterpart from CIFAR10 dataset
    """
    device = torch.device("cuda")
    # Path to the model pre-trained on CIFAR100 and linear-prrobed on CIFAR10
    ckpt_path = "experiments/bs_128_ds_cifar100-cifar10_eps_10_lr_0.001_lrs_cosine_tf_method_lp/models/cifar100-cifar10_lp.pth" 
    epsilon=[16 / 255]
    
    ## Dataloader
    dataloader = get_dataloader(
        '',
        'cifar10',
        True,
        batch_size=1,
        shuffle=True,
    )

    ## Constructs model
    model = get_model().to(device)
    model = load_model(model, ckpt_path)
    model.eval()
    # feature extractor model
    modules = list(model.children())[:-1]
    model_feat = nn.Sequential(*modules)
    model.eval()
    fmodel_feat = fb.PyTorchModel(model_feat, bounds=(-1.0, 1.0))
    
    ## Attack
    l_inf_pgd = fb.attacks.LinfPGD(steps=20) 
    
    ## Construct Adversarials
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    inputs = inputs.resize(inputs.size(0), 3, 32, 32)
    with torch.no_grad(): frep = model_feat(inputs.to(device))
    criterion = fb.criteria.NormDiff(frep)
    _, adv_batch, success = l_inf_pgd(fmodel_feat, inputs, criterion=criterion, epsilons=epsilon)
    adv_batch[0] = adv_batch[0].resize(inputs.size(0), 3, 32, 32)

    ## Save 
    p = transforms.Compose([transforms.Resize((64,64))])
    save_image(p(inputs[0]), 'original-CIFAR10.png')
    save_image(p(adv_batch[0][0]), 'adversarial-CIFAR10.png')
    print("Save Successful!!")