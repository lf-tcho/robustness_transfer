# Setup
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
python dsprites_lp.py --target_latent orientation --train_only_fclayer False
```

## Evaluating on dsprites

Attack on the logits (output from last layer)
```bash
python dsprites_theory_analysis.py --target_latent orientation
```

Attack on the representation function.
```bash
python dsprites_theory_analysis-attack_feat2.py --target_latent orientation
```

## CIFAR Dataset
### Linear Probing / Finetuning on CIFAR10

Take a robustly pretrained CIFAR100 model and linear probe/finetune on CIFAR10 dataset.
```bash
python cifar_lp.py --train_only_fclayer False
```

### Evaluating on CIFAR10

Attack on the logits (output from last layer).
```bash
python cifar_theory_analysis.py 
```

Attack on the representation function.
```bash
python cifar_theory_analysis-attack_feat2.py
```
