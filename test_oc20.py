# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from E2GNN import E2GNN
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
import numpy as np
from utils import *
from sklearn.metrics import mean_absolute_error
from functools import partial
from torch.utils.data import DataLoader
from ema import EMAHelper
from collections import defaultdict
import pandas as pd
import argparse

import warnings
warnings.filterwarnings("ignore")

# %%
def val(model, dataloader, device):
    model.eval()

    pred_energy_list = []
    pred_force_list = []
    label_energy_list = []
    label_force_list = []
    fixed_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred_energy, pred_force = model(data)
            label_energy, label_force, fixed = data.y, data.force, data.fixed

            pred_energy_list.append(pred_energy.detach().cpu().numpy())
            label_energy_list.append(label_energy.detach().cpu().numpy())

            pred_force_list.append(pred_force.detach().cpu().numpy())
            label_force_list.append(label_force.detach().cpu().numpy())

            fixed_list.append(fixed.detach().cpu().numpy())

    pred_energy = np.concatenate(pred_energy_list, axis=0)
    label_energy = np.concatenate(label_energy_list, axis=0)

    pred_force = np.concatenate(pred_force_list, axis=0)
    label_force = np.concatenate(label_force_list, axis=0)

    fixed = np.concatenate(fixed_list, axis=0)
    mask = (fixed == 0)
    pred_force = pred_force[mask]
    label_force = label_force[mask]

    mae_energy = mean_absolute_error(pred_energy, label_energy)
    mae_force = mean_absolute_error(pred_force, label_force)

    model.train()

    return mae_energy, mae_force

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--data_type', type=str, choices=['50K', '200K'], default='50K', help='type of data')
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir such as ./wandb/run-20231031_144315-E2GNN_20231031_144314', required=True)
    parser.add_argument('--model_type', type=str, choices=['E2GNN', 'vanilla', 'vanilla_nmu'], default='E2GNN', help='type of data')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    args = parser.parse_args()
    data_root = args.data_root
    data_type = args.data_type
    model_dir = args.model_dir
    model_type = args.model_type
    batch_size = args.batch_size 

    model_path = os.path.join(model_dir, 'files', 'model.pt')

    device = torch.device('cuda:0')
    if model_type == 'vanilla':
        from vanilla import E2GNN
        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=True, otf_graph=False, num_elements=118).to(device)
    elif model_type == 'vanilla_nmu':
        from vanilla_nmu import E2GNN
        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=True, otf_graph=False, num_elements=118).to(device)
    else:
        from E2GNN import E2GNN
        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=True, otf_graph=False, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    val_root = '/scratch/yangzd/materials/data/oc20/all'
    for val_mode in ['val_ood_ads', 'val_ood_cat', 'val_ood_both', 'val_id']:
        performance_dict = defaultdict(list)
        valid_set = TrajectoryLmdbDataset({"src": os.path.join(val_root, val_mode), "split":False})
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, otf_graph=False), num_workers=4)

        valid_mae_energy, valid_mae_force = val(model, valid_loader, device)

        performance_dict[f'E2GNN_{val_mode}'].append(valid_mae_energy)
        performance_dict[f'E2GNN_{val_mode}'].append(valid_mae_force)

        df = pd.DataFrame(performance_dict)
        df.to_csv(f'{model_type}_{data_type}_{val_mode}.csv', index=False)

        print(f"finish {val_mode}")

# %%

