## Experiments

Transfer learning used for sections 5.2 and 5.3\
Linear probing with cifar100 pre-trained model, run experiments_configs.lp_experiment.py\

bs_128_ds_cifar10_eps_20_lr_0.01_lrs_cosine_tf_method_lp\
bs_128_ds_fashion_eps_20_lr_0.01_lrs_cosine_tf_method_lp\
bs_128_ds_intel_image_eps_20_lr_0.01_lrs_cosine_tf_method_lp\

bs = batch size, ds = dataset, eps = epochs, lr = learning rate, lrs = learning rate schedule, 
tf method lp = linear probing\

Linear probing with imagenet pre-trained model, run experiments_configs.imagenet_experiment.py\

__imagenet_bs_32_ds_cifar10_eps_10_lr_0.001_lrs_cosine_tf_method_lp\
__imagenet_bs_32_ds_fashion_eps_10_lr_0.001_lrs_cosine_tf_method_lp\
__imagenet_bs_32_ds_intel_image_eps_10_lr_0.001_lrs_cosine_tf_method_lp\

Calculate most values for theory with cifar100_theory_analysis and attack on the representation function 
(preliminary version) with cifar100_adv_representation_analysis. Similar for ImageNet.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Folder structure

- src/: Contains main code for training and evaluation 
- experiment_configs/: Contains experiment configurations using the src code 

For each experiment define a experiment config with specific train code (optimizer, dataloader, model, etc.). 
Trainer class handles generic training. Evaluator class evaluates the model of a specific experiment 
(adverserial attacks, etc.). 