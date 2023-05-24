***

## Files
- `dsprites_adv_train.py` pre-trains a model from scratch on dSprites Dataset.
- `dsprites_lp.py` linear probes/ finetunes the pre-trained model (saved after running the above script) on a specific target latent factor (scale, orientation, posX, posY).
- `dsprites_theory_analysis-attack_feat2.py` calculates the LHS and RHS of the Lemma 1 (see paper) using the linear probed model (saved after running the above script).

# Setup
### Install dependencies

```bash
pip install -r requirements.txt
```

## Experiments - dSprites Dataset - Table 4 (see paper)
### Training (from scratch) on dsprites

```bash
python dsprites_adv_train.py --target_latent orientation
```

### Linear Probing / Finetuning on dsprites
Run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--target_latent` argument.
```bash
python dsprites_lp.py --target_latent orientation --train_only_fclayer False
```

### Calculate LHS and RHS for Lemma 1 (see paper).

Below are few variations which we tried to reporduce Table 4.

- **Adversarial - Linf PGD**: run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--finetune_target_latent` argument.
```bash
python dsprites_theory_analysis-attack_feat2.py --pretrain_model_type robust --pretrain_target_latent orientation --finetune_target_latent scale --attack_type linf_pgd --model_type lp 
```

- **Standard - Linf PGD**: run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--finetune_target_latent` argument.
```bash
python dsprites_theory_analysis-attack_feat2.py --pretrain_model_type clean --pretrain_target_latent orientation --finetune_target_latent scale --attack_type linf_pgd --model_type lp 
```

- **Random - Linf PGD**: run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--finetune_target_latent` argument.
```bash
python dsprites_theory_analysis-attack_feat2.py --pretrain_model_type random --pretrain_target_latent orientation --finetune_target_latent scale --attack_type linf_pgd --model_type lp 
```

- **Adversarial - Linf PGD**: run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--finetune_target_latent` argument.
```bash
python dsprites_theory_analysis-attack_feat2.py --pretrain_model_type robust --pretrain_target_latent orientation --finetune_target_latent scale --attack_type l2_pgd --model_type lp 
```

- **Standard - L2 PGD**: run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--finetune_target_latent` argument.
```bash
python dsprites_theory_analysis-attack_feat2.py --pretrain_model_type clean --pretrain_target_latent orientation --finetune_target_latent scale --attack_type l2_pgd --model_type lp 
```

- **Random - L2 PGD**: run the below command separately for all the target latents - scale, orientation, posX, posY specifying it in the `--finetune_target_latent` argument.
```bash
python dsprites_theory_analysis-attack_feat2.py --pretrain_model_type random --pretrain_target_latent orientation --finetune_target_latent scale --attack_type l2_pgd --model_type lp 
```
