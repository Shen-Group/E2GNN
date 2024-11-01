# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from pathlib import Path
import json
import argparse

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import mean_absolute_error

from ase import units
from ase.md.langevin import Langevin
from ase_utils import data_to_atoms, MDCalculator, Simulator

from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from E2GNN import E2GNN
from utils import RemoveMean
from ema import EMAHelper

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

    return mae_energy, mae_force

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir such as ./wandb/run-20231031_144315-E2GNN_20231031_144314/', required=True)
    parser.add_argument('--timestep', type=float, default=0.5, help='This defines the time interval for each step of the simulation, in femtoseconds (fs)')
    parser.add_argument('--temperature_K', type=float, default=300., help='This sets the target temperature for the Langevin thermostat, in Kelvin')
    parser.add_argument('--friction', type=float, default=8, help='This parameter is specific to the Langevin integrator and controls the strength of the friction (or damping) term')
    parser.add_argument('--save_frequency', type=int, default=100, help='This parameter defines how often the data from the simulation is saved')
    parser.add_argument('--T_init', type=float, default=300., help='This parameter sets the initial temperature of the system at the start of the simulation, measured in Kelvin')
    parser.add_argument('--steps', type=int, default=100000, help='This is the total number of steps the simulation will run.')

    args = parser.parse_args()
    data_root = args.data_root
    model_dir = args.model_dir
    timestep = args.timestep
    temperature_K = args.temperature_K
    friction = args.friction
    save_frequency = args.save_frequency
    T_init = args.T_init
    steps = args.steps

    model_path = os.path.join(model_dir, 'files', 'model.pt')

    test_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'test')})
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=partial(collate_fn, otf_graph=True), num_workers=4)
    
    mean = torch.load(os.path.join(data_root, 'data_stat.pkl'))
    normalizer = RemoveMean(mean)

    device = torch.device('cuda:0')
    model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=5.0, max_neighbors=50, use_pbc=True, otf_graph=True, num_elements=118)
    
    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path)) # load a trained model
    ema_helper.ema(model)
    
    model = model.to(device)

    # evaluate the model performance
    mae_energy, mae_force = val(model, test_loader, normalizer, device=device)
    
    md_dir = os.path.join(model_dir, "md")
    (Path(md_dir)).mkdir(parents=True, exist_ok=True)

    results_path = os.path.join(md_dir, 'results.json')
    with open(results_path, 'w') as json_file:
        json.dump({"energy": mae_energy.item(), "forces": mae_force.item()}, json_file, indent=4) # Writing to a file

    # run MD simulation
    init_idx = random.randint(0, len(test_set))
    init_data = test_set[init_idx][0] # random select a configuration as the starting point
    calculator = MDCalculator(model, normalizer, device) # use E2GNN as a surrogate ML models to approximate energy and forces at each iteration
    atoms = data_to_atoms(init_data)
    atoms.set_calculator(calculator)

    integrator = Langevin(atoms, timestep*units.fs, temperature_K=temperature_K, friction=friction)
    simulator = Simulator(atoms, integrator, T_init=T_init, 
                            start_time=0,
                            save_dir=md_dir, 
                            save_frequency=save_frequency)
    # start simulation, results are stored in md_dir
    early_stop, step = simulator.run(steps)


