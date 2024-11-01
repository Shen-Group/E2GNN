# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from leignn import LEIGNN
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
import numpy as np
from sklearn.metrics import mean_absolute_error
from functools import partial
from torch.utils.data import DataLoader
from utils import RemoveMean
from ema import EMAHelper
import argparse

import warnings
warnings.filterwarnings("ignore")

# %%
def val(model, dataloader, normalizer, device):
    model.eval()

    pred_energy_list = []
    pred_force_list = []
    label_energy_list = []
    label_force_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred_energy, pred_force = model(data)
            pred_energy = normalizer.denorm(pred_energy)
            label_energy, label_force = data.y, data.force

            pred_energy_list.append(pred_energy.detach().cpu().numpy())
            label_energy_list.append(label_energy.detach().cpu().numpy())

            pred_force_list.append(pred_force.detach().cpu().numpy())
            label_force_list.append(label_force.detach().cpu().numpy())
            
    pred_energy = np.concatenate(pred_energy_list, axis=0)
    label_energy = np.concatenate(label_energy_list, axis=0)

    pred_force = np.concatenate(pred_force_list, axis=0)
    label_force = np.concatenate(label_force_list, axis=0)

    mae_energy = mean_absolute_error(pred_energy, label_energy)
    mae_force = mean_absolute_error(pred_force, label_force)

    model.train()

    return mae_energy, mae_force

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir such as ./wandb/run-20231031_144315-LEIGNN_20231031_144314', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')

    args = parser.parse_args()
    data_root = args.data_root
    model_dir = args.model_dir

    model_path = os.path.join(model_dir, 'files', 'model.pt')

    valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'valid_dataset')})
    test_within_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'test_within_dataset')})
    test_other_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'test_other_dataset')})
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn=partial(collate_fn, otf_graph=True), num_workers=4)
    test_within_loader = DataLoader(test_within_set, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, otf_graph=True), num_workers=4)
    test_other_loader = DataLoader(test_other_set, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, otf_graph=True), num_workers=4)

    mean = torch.load(os.path.join(data_root, 'data_stat.pkl'))
    normalizer = RemoveMean(mean)

    device = torch.device('cuda:0')
    model = LEIGNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=False, otf_graph=True).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    valid_mae_energy, valid_mae_force = val(model, valid_loader, normalizer, device)
    test_within_mae_energy, test_within_mae_force = val(model, test_within_loader, normalizer, device)
    test_other_mae_energy, test_other_mae_force = val(model, test_other_loader, normalizer, device)

    print("valid_mae_energy: ", valid_mae_energy)
    print("valid_mae_force: ", valid_mae_force)
    print("test_within_mae_energy: ", test_within_mae_energy)
    print("test_within_mae_force: ", test_within_mae_force)
    print("test_other_mae_energy: ", test_other_mae_energy)
    print("test_other_mae_force: ", test_other_mae_force)
    
# %%