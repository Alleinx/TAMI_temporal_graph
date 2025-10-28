# TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction

## Overview
This is the official implementation of the NeurIPS 2025 paper: [*TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction*](https://arxiv.org/abs/2510.23577). 

Our code is developed based on the codebase of the paper: [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047). We made modifications to four files: **models/modules.py**, **train_link_prediction.py**, **evaluate_link_prediction.py**, and **evaluate_model_utils.py**.

## Environments
[PyTorch 1.8.1](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[tqdm](https://github.com/tqdm/tqdm), and 
[tabulate](https://github.com/astanin/python-tabulate)

## Benchmark Datasets and Preprocessing
Most of our experiments are conducted on 13 datasets from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg), including Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, Can. Parl., US Legis., UN Trade, UN Vote, and Contact. 

## Baseline Link Prediction Methods
Nine popular link prediction methods for continuous-time dynamic graphs are included in our experiment:
[JODIE](https://dl.acm.org/doi/10.1145/3292500.3330895), 
[DyRep](https://openreview.net/forum?id=HyePrhR5KX), 
[TGAT](https://openreview.net/forum?id=rJeW1yHYwH), 
[TGN](https://arxiv.org/abs/2006.10637), 
[CAWN](https://openreview.net/forum?id=KYPz4YsCPj), 
[EdgeBank](https://openreview.net/forum?id=1GVpwr2Tfdg), 
[TCL](https://arxiv.org/abs/2105.07944), 
[GraphMixer](https://openreview.net/forum?id=ayPPc0SyLv1), and
[DyGFormer](https://arxiv.org/abs/2303.13047).

## Executing Scripts

### Scripts for Dynamic Link Prediction
If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.
#### Model Training
* Example of training *GraphMixer* on *Wikipedia* dataset:
```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name GraphMixer --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to train *GraphMixer* on *Wikipedia* dataset, run
```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --num_runs 5 --gpu 0
```
#### Model Evaluation
* Example of evaluating *GraphMixer* on the *Wikipedia* dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name GraphMixer --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to evaluate *GraphMixer* on *Wikipedia* dataset, run
```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --num_runs 5 --gpu 0
```
## Citation
Will be added later.
