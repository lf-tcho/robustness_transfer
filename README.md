## Setup
### Install dependencies

```bash
pip install -r requirements.txt
```

## DSprites Dataset
### Training (from scratch) on dsprites

```bash
python dsprites_adv_train.py --target_latent orientation
```

### Linear Probing / Finetuning on dsprites

```bash
python dsprites_lp.py --target_latent orientation
```

### Evaluating on dsprites (attack on the representation function)

```bash
python dsprites_theory_analysis-attack_feat2.py --target_latent orientation
```

## CIFAR Dataset
### Linear Probing / Finetuning on CIFAR10

```bash
python cifar_lp.py
```

### Evaluating on CIFAR10 (attack on the representation function)

```bash
python cifar_theory_analysis-attack_feat2.py 
```
