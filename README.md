# E2GNN

## Dataset
We use ISO17, H2O, and CH4 as examples to illustrate how to use E2GNN.

- **ISO17 Dataset** [1]: Available at [ISO17](http://quantum-machine.org/datasets/).
- **OC20 Dataset** [2]: Available at [OC20 (Train and Val)](https://zenodo.org/records/12508546) and [OC20 (Test)](https://zenodo.org/records/11375993).
- **H2O and CH4 Datasets**: Available at [H2O and CH4](https://zenodo.org/records/10208201).
- **LiPS**: Available at [LiPS](https://zenodo.org/records/7196767).

## Requirements
Required Python packages include:  
- `ase==3.22.1`
- `config==0.5.1`
- `lmdb==1.4.1`
- `matplotlib==3.7.2`
- `numpy==1.24.4`
- `pandas==2.1.3`
- `pymatgen==2023.5.10`
- `scikit_learn==1.3.0`
- `scipy==1.11.4`
- `torch==1.13.1`
- `torch_geometric==2.2.0`
- `torch_scatter==2.1.0`
- `tqdm==4.66.1`

Alternatively, install the environment using the provided YAML file at `./environment/environment.yaml`.

## Logger
For logging, we recommend using wandb. More details are available at https://wandb.ai/. Training logs and trained models are stored in the `./wandb` directory.

## Step-by-Step Guide

### Data Preprocessing
Download the data from [ISO17 + H2O + CH4](https://zenodo.org/records/10208201), [OC20 (Train and Val)](https://zenodo.org/records/12508546), [OC20 (Test)](https://zenodo.org/records/11375993) and [LiPS](https://zenodo.org/records/7196767).
The downloaded data are preprocessed by default. If you wish to preprocess them from scratch, run:
- `python preprocess_iso17.py --data_root /path/to/iso17 --num_workers 8` for the ISO17 dataset.
- `python preprocess_md.py --data_root /path/to/CH4 --num_workers 8` for the CH4 dataset.
- `python preprocess_md.py --data_root /path/to/H2O --num_workers 8` for the H2O dataset.

Replace `/path/to/` with your directory paths.

### Train the Model
To train E2GNN, run:
- `python train_iso17.py --data_root /path/to/iso17 --num_workers 4` for ISO17.
- `python train_oc20.py --data_root /path/to/oc20/200k --data_type 50K --model_type E2GNN --num_workers 4 --batch_size 32` for OC20-50K and `python train_oc20.py --data_root /path/to/oc20/200k --data_type 200K --model_type E2GNN --num_workers 4 --batch_size 32` for OC20-200K.
- `python train_md.py --data_root /path/to/CH4 --systems CH4 --num_workers 4` for CH4.
- `python train_md.py --data_root /path/to/H2O --systems H2O --num_workers 4` for H2O.

Replace `/path/to/` with your directory paths.

### Ablation study
To perform the ablation study, use the following commands:
- `python train_oc20.py --data_root /path/to/oc20/200k --data_type 50K --model_type vanilla --num_workers 4 --batch_size 32` for OC20-50K and `python train_oc20.py --data_root /path/to/oc20/200k --data_type 200K --model_type E2GNN --num_workers 4 --batch_size 32` for Vanilla.
- `python train_oc20.py --data_root /path/to/oc20/200k --data_type 50K --model_type vanilla_nmu --num_workers 4 --batch_size 32` for OC20-50K and `python train_oc20.py --data_root /path/to/oc20/200k --data_type 200K --model_type E2GNN --num_workers 4 --batch_size 32` for Vanilla + NMU.

To test the case for 200K, simply replace --data_type 50K with --data_type 200K in the commands above.

### Test the Model
To test E2GNN on ISO17, run:
- `python test_iso17.py --data_root /path/to/iso17 --model_dir ./wandb/run-20231031_144315-E2GNN_20231031_144314/`
To test E2GNN on oc20, run:
- `python test_oc20.py --data_root /path/to/oc20/200k --data_type 50K --model_dir ./wandb/run-20231031_144315-E2GNN_20231031_144314/ --model_type E2GNN --batch_size 32`

Replace `/path/to/` and `./wandb/run-20231031_144315-E2GNN_20231031_144314/` with your directory path.

### MD Simulations
After training E2GNN on the LiPS, H2O, and CH4 datasets, run MD simulations with the following commands:
#### H20
- `python simulate_md.py --data_root /path/to/H2O --model_dir ./wandb/run-20231124_233309-E2GNN_H2O_20231124_233308`
Replace `/path/to/` and `./wandb/run-20231124_233309-E2GNN_H2O_20231124_233308` with your directory path.  
Evaluate and visualize MD simulation results using `performance_H2O.ipynb`

#### CH4
- `python simulate_md.py --data_root /path/to/CH4 --model_dir ./wandb/run-20231124_233309-E2GNN_CH4_20231124_233308`
Evaluate and visualize MD simulation results using `performance_CH4.ipynb`

#### LiPS
- `python simulate_lips.py --data_root /path/to/lips/20k --model_dir ./wandb/run-20240717_202344-E2GNN_LiPS_20240717_202343`  
Evaluate and visualize MD simulation results using `performance_lips.py`

## Acknowledgements
Some part of code in this project were adapted from [OCP](https://github.com/Open-Catalyst-Project/ocp) and [MDsim](https://github.com/kyonofx/MDsim). We gratefully acknowledge the contributions from these sources.
## Reference
[1] Schütt K, Kindermans P J, Sauceda Felix H E, et al. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions." Advances in Neural Information Processing Systems, 2017, 30.
