# On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2208.03835-b31b1b.svg)](https://arxiv.org/abs/2208.03835)
[![NeurIPS 2023](https://img.shields.io/badge/NeurIPS-2023-blue.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9801626a6ffaf6664af1e983dbd0094-Abstract-Conference.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for the NeurIPS 2023 paper **"On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks"** by Laura Fee Nern, Harsh Raj, Maurice André Georgi, and Yash Sharma.

📄 [Paper (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9801626a6ffaf6664af1e983dbd0094-Abstract-Conference.html) &nbsp;|&nbsp;
📄 [arXiv](https://arxiv.org/abs/2208.03835) &nbsp;|&nbsp;
📄 [OpenReview](https://openreview.net/forum?id=D8nAMRRCLS)

## Abstract

> As large-scale training regimes have gained popularity, the use of pretrained models for downstream tasks has become common practice in machine learning. While pretraining has been shown to enhance the performance of models in practice, the transfer of robustness properties from pretraining to downstream tasks remains poorly understood. In this study, we demonstrate that the robustness of a linear predictor on downstream tasks can be constrained by the robustness of its underlying representation, regardless of the protocol used for pretraining. We prove (i) a bound on the loss that holds independent of any downstream task, as well as (ii) a criterion for robust classification in particular. We validate our theoretical results in practical applications, show how our results can be used for calibrating expectations of downstream robustness, and when our results are useful for optimal transfer learning. Taken together, our results offer an initial step towards characterizing the requirements of the representation function for reliable post-adaptation performance.

## Overview

This repository contains the code used to run the transfer-learning and robustness-evaluation experiments in the paper, including:

- **Linear probing (LP), fine-tuning (FT), and LP-FT** transfer protocols on top of adversarially pretrained CIFAR-100 and ImageNet backbones (via [RobustBench](https://github.com/RobustBench/robustbench)), transferred to downstream datasets including CIFAR-10, CIFAR-100, dSprites, Fashion-MNIST, Weather, and Intel Image Classification.
- **Robust accuracy evaluation** under $\ell_\infty$ PGD attacks (via [Foolbox](https://github.com/bethgelab/foolbox)).
- **Representation-level analyses** (adversarial representation shift, "Learning without Forgetting"-style regularization) used to empirically validate the paper's theoretical bounds.

## Installation

```bash
git clone https://github.com/lf-tcho/robustness_transfer.git
cd robustness_transfer
pip install -r requirements.txt
```

Requires PyTorch, [RobustBench](https://github.com/RobustBench/robustbench), [Foolbox](https://github.com/bethgelab/foolbox), and TensorBoard (see [requirements.txt](requirements.txt)).

> **Note:** modules use relative imports rooted at the `robustness_transfer` package (e.g. `from ..src.dataloader import ...`), so scripts must be run as modules **from the parent directory of the clone** (`python -m robustness_transfer.<...>`), not from inside `robustness_transfer/` itself — see [Usage](#usage).

## Repository structure

```
.
├── src/                      # Core library
│   ├── experiment.py         # Abstract Experiment base class
│   ├── trainer.py            # Trainer / TrainerLwF training loops
│   ├── evaluator.py          # Clean & adversarial (PGD) evaluation
│   ├── dataloader.py         # Dataset loading utilities
│   ├── models.py             # Model wrappers/architectures
│   ├── transforms.py         # Custom image transforms
│   └── utils.py              # Misc helpers
└── experiments_configs/      # One config per experiment (see Usage for how each is run)
    ├── lp_experiment.py                       # CIFAR-10/100 linear probing / fine-tuning
    ├── imagenet_experiment.py                 # ImageNet linear probing / fine-tuning
    ├── cifar100_analysis.py                   # CIFAR-100 robustness analysis
    ├── imagenet_analysis.py                   # ImageNet robustness analysis
    ├── cifar100_theory_analysis.py            # Empirical validation of theoretical bounds (CIFAR-100)
    ├── imagenet_theory_analysis.py            # Empirical validation of theoretical bounds (ImageNet)
    ├── dsprites_theory_analysis.py            # Empirical validation of theoretical bounds (dSprites)
    ├── cifar100_adv_representation_analysis.py    # Adversarial representation-shift analysis (CIFAR-100)
    └── imagenet_adv_representation_analysis.py    # Adversarial representation-shift analysis (ImageNet)
```

Each experiment config defines the model, optimizer, and dataloaders for that setting; `Trainer` handles the generic training loop and `Evaluator` computes clean/robust accuracy.

## Usage

Run experiments as modules **from the directory containing the cloned repo** (i.e. one level above `robustness_transfer/`), so the package's relative imports resolve correctly:

```bash
cd ..   # parent directory of the robustness_transfer/ clone
```

`lp_experiment.py` (CIFAR transfer) and `imagenet_experiment.py` (ImageNet transfer) are CLI-configurable. For example, to linear-probe (`lp`) a robustly pretrained WideResNet, transferring from CIFAR-100 to CIFAR-10:

```bash
python -m robustness_transfer.experiments_configs.lp_experiment \
    --batch_size 128 \
    --epochs 20 \
    --learning_rate 0.001 \
    --tf_method lp \
    --dataset_name cifar10 \
    --device cuda
```

Available transfer methods (`--tf_method`) include `lp` (linear probing), `lp_ft` (LP then full fine-tuning), and per-layer freezing variants (`block1`, `block1_lp`, etc.). Checkpoints, TensorBoard logs, and evaluation metrics are written to `robustness_transfer/experiments/<experiment_name>/`. Run `python -m robustness_transfer.experiments_configs.lp_experiment --help` (or `...imagenet_experiment --help`) for the full list of options.

The remaining `experiments_configs/*.py` scripts (`cifar100_analysis.py`, `imagenet_analysis.py`, `*_theory_analysis.py`, `*_adv_representation_analysis.py`) are **not** CLI-configurable — each hardcodes its experiment config (checkpoint name, dataset, target latent, etc.) in `main()`. To change their parameters, edit the values in the script directly, then run e.g.:

```bash
python -m robustness_transfer.experiments_configs.cifar100_theory_analysis
```

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{NEURIPS2023_b9801626,
 author = {Nern, Laura F. and Raj, Harsh and Georgi, Maurice Andr\'{e} and Sharma, Yash},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {59206--59226},
 publisher = {Curran Associates, Inc.},
 title = {On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/b9801626a6ffaf6664af1e983dbd0094-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

## Acknowledgements

This code was originally developed from [MauGeo/dmtml](https://github.com/MauGeo/dmtml), a course project repository, and builds on [RobustBench](https://github.com/RobustBench/robustbench) for pretrained robust model checkpoints and architectures, and [Foolbox](https://github.com/bethgelab/foolbox) for adversarial attacks.

## License

This project is licensed under the [MIT License](LICENSE).
