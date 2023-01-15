## Th3.2 on dsprites(regression)

## Setup
## Install dependencies

```bash
pip install -r requirements.txt
```

## Training (from scratch) on dsprites

```bash
python dsprites_adv_train.py --target_latent orientation
```

## Linear Probing / Finetuning on dsprites

```bash
python dsprites_lp.py --target_latent orientation
```

## Evaluating on dsprites

```bash
python dsprites_theory_analysis.py --target_latent orientation
```