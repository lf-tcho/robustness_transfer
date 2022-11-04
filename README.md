## Install dependencies

```bash
pip install -r requirements.txt
```

## Folder structure

- src/: Contains main code for training and evaluation 
- experiment_configs/: Contains experiment configurations using the src code 

For each experiment define a experiment config with specific train code (optimizer, dataloader, model, etc.). 
Trainer class handles generic training. Evaluator class evaluates the model of a specific experiment (adverserial attacks, etc.). 