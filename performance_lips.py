# %%
import os
import json
import torch
import numpy as np
import itertools
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ase.io import read, Trajectory
import pickle

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_hr(traj, bins):
    """
    compute h(r) for MD17 simulations.
    traj: T x N_atoms x 3
    """
    pdist = torch.cdist(traj, traj).flatten()
    hist, _ = np.histogram(pdist[:].flatten().numpy(), bins, density=True)
    return hist

"""
functions for loading simulated trajectories and computing observables.
"""
def get_thermo(filename):
    """
    read thermo logs.
    """
    with open(filename, 'r') as f:
        thermo = f.read().splitlines()
        sim_time, Et, Ep, Ek, T = [], [], [], [], []
        for i in range(1, len(thermo)):
            try:
                t, Etot, Epot, Ekin, Temp = [float(x) for x in thermo[i].split(' ') if x]
                sim_time.append(t)
                Et.append(Etot)
                Ep.append(Epot)
                Ek.append(Ekin)
                T.append(Temp)
            except:
                sim_time, Et, Ep, Ek, T = [], [], [], [], []
    thermo = {
        'time': sim_time,
        'Et': Et,
        'Ep': Ep,
        'Ek': Ek,
        'T': T
    }
    return thermo

def get_test_metrics(md_dir):
    """
    read test metrics such as force error.
    """
    run_metrics = {}
    with open(md_dir / 'test_metric.json', 'r') as f:
        test_metric = json.load(f)
        
        if 'mae_f' in test_metric:
            fmae = test_metric['mae_f']
            run_metrics['fmae'] = fmae
        elif 'f_mae' in test_metric:
            fmae = test_metric['f_mae']
            run_metrics['fmae'] = fmae
        elif 'forces_mae' in test_metric:
            fmae = test_metric['forces_mae']['metric']
            run_metrics['fmae'] = fmae
            
        if 'mae_e' in test_metric:
            emae = test_metric['mae_e']
            run_metrics['emae'] = emae
        elif 'e_mae' in test_metric:
            emae = test_metric['e_mae']
            run_metrics['emae'] = emae
        elif 'energy_mae' in test_metric:
            emae = test_metric['energy_mae']['metric']
            run_metrics['emae'] = emae
            
        if 'num_params' in test_metric:
            run_metrics['n_params'] = test_metric['num_params']
        if 'running_time' in test_metric:
            run_metrics['running_time'] = test_metric['running_time']
    return run_metrics

def mae(x, y, factor):
    return np.abs(x-y).mean() * factor

def distance_pbc(x0, x1, lattices):
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1,1,3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta ** 2).sum(dim=-1))

def get_diffusivity_traj(pos_seq, dilation=1):
    """
    Input: B x N x T x 3
    Output: B x T
    """
    # substract CoM
    bsize, time_steps = pos_seq.shape[0], pos_seq.shape[2]
    pos_seq = pos_seq - pos_seq.mean(1, keepdims=True)
    msd = (pos_seq[:, :, 1:] - pos_seq[:, :, 0].unsqueeze(2)).pow(2).sum(dim=-1).mean(dim=1)
    diff = msd / (torch.arange(1, time_steps)*dilation) / 6
    return diff.view(bsize, time_steps-1)

def get_smoothed_diff(xyz):
    seq_len = xyz.shape[0] - 1
    diff = torch.zeros(seq_len)
    for i in range(seq_len):
        diff[:seq_len-i] += get_diffusivity_traj(xyz[i:].transpose(0, 1).unsqueeze(0)).flatten()
    diff = diff / torch.flip(torch.arange(seq_len),dims=[0])
    return diff

# %%
def compute_image_flag(cell, fcoord1, fcoord2):
    supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3))).to(cell.device)
    fcoords = fcoord2[:, None] + supercells
    coords = fcoords @ cell
    coord1 = fcoord1 @ cell
    dists = torch.cdist(coord1[:, None], coords).squeeze()
    image = dists.argmin(dim=-1)
    return supercells[image].long()

def frac2cart(fcoord, cell):
    return fcoord @ cell

def cart2frac(coord, cell):
    invcell = torch.linalg.inv(cell)
    return coord @ invcell

# the source data is in wrapped coordinates. need to unwrap it for computing diffusivity.
def unwrap(pos0, pos1, cell):
    fcoords1 = cart2frac(pos0, cell)
    fcoords2 = cart2frac(pos1, cell)
    flags = compute_image_flag(cell, fcoords1, fcoords2)
    remapped_frac_coords = cart2frac(pos1, cell) + flags
    return frac2cart(remapped_frac_coords, cell)

# different from previous functions, now needs to deal with non-cubic cells. 
def compute_distance_matrix_batch(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells+1, 1).to(cell.device)
    combos = torch.stack(
        torch.meshgrid(pos, pos, pos, indexing='xy')
            ).permute(3, 2, 1, 0).reshape(-1, 3).to(cell.device)
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    # NxNxCells distance array
    shifted = cart_coords.unsqueeze(2) + shifts.unsqueeze(0).unsqueeze(0)
    dist = cart_coords.unsqueeze(2).unsqueeze(2) - shifted.unsqueeze(1)
    dist = dist.pow(2).sum(dim=-1).sqrt()
    # But we want only min
    distance_matrix = dist.min(dim=-1)[0]
    return distance_matrix

def get_lips_rdf(data_seq, lattices, bins, device='cpu'):
    data_seq = data_seq.to(device).float()
    lattices = lattices.to(device).float()
    
    lattice_np = lattices.cpu().numpy()
    volume = float(abs(np.dot(np.cross(lattice_np[0], lattice_np[1]), lattice_np[2])))
    data_pdist = compute_distance_matrix_batch(lattices, data_seq)

    data_pdist = data_pdist.flatten().cpu().numpy()
    data_shape = data_pdist.shape[0]

    data_pdist = data_pdist[data_pdist != 0]
    data_hist, _ = np.histogram(data_pdist, bins)

    rho_data = data_shape / volume
    Z_data = rho_data * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rdf = data_hist / Z_data
        
    return rdf

def load_run(md_dir, atomic_numbers, cell, xlim, bins, stability_threshold, gt_rdf, gt_diff):
    if not isinstance(md_dir, Path):
        md_dir = Path(md_dir)
        
    model_name = md_dir.parts[-2]
    seed = md_dir.parts[-1][-1]
    run = {'name': (model_name + f'_seed_{seed}')}

    run['traj'] = Trajectory(md_dir / 'atoms.traj')
    run['traj'] = torch.from_numpy(np.stack([run['traj'][i].positions 
                                                  for i in range(len(run['traj']))]))
    run['thermo'] = get_thermo( md_dir / 'thermo.log')

    md_time = np.array(run['thermo']['time'])
    T = np.array(run['thermo']['T']) 
    collapse_pt = len(T)
    for i in (range(1, len(T)-rdf_check_interval)):
        timerange = torch.arange(i, i + rdf_check_interval)
        current_rdf = get_lips_rdf(run['traj'][timerange], cell, bins)
        rdf_mae = mae(current_rdf, gt_rdf, xlim)
        if rdf_mae > stability_threshold:
            collapse_pt = i
            break

    run['collapse_pt'] = collapse_pt 

    run['rdf'] = get_lips_rdf(run['traj'][:collapse_pt], cell, bins)
    run['rdf_error'] = mae(run['rdf'], gt_rdf, xlim)

    if collapse_pt > 3200:
        # removing the first 5 ps for equilibrium. use the diffusivity at 40 ps as a convergence value.
        # some random error is unavoidable with 50-ps reference simulations.
        diff = get_smoothed_diff(run['traj'][400:collapse_pt:4, atomic_numbers == 3])
        run['diffusivity'] = diff[700] * 20 * 1e-8
        run['end_diff'] = float(run['diffusivity'])
        run['diff_error'] = np.abs(float(run['diffusivity']) - float(gt_diff[700]))
    else:
        run['diffusivity'] = None
        run['end_diff'] = np.inf
        run['diff_error'] = np.inf

    # load test metrics
    if (md_dir / 'test_metric.json').exists():
        test_metrics = get_test_metrics(md_dir)
        run.update(test_metrics)

    return run

def force_mag_per_atom(atoms):
    forces_mag_dict = defaultdict(list)
    positions_list = []
    for atom in atoms:
        forces = atom.get_forces()
        atomic_numbers = atom.get_atomic_numbers()

        forces_mag = np.linalg.norm(forces, axis=-1)
        forces_mag3 = forces_mag[atomic_numbers == 3]
        forces_mag15 = forces_mag[atomic_numbers == 15]
        forces_mag16 = forces_mag[atomic_numbers == 16]

        forces_mag_dict['atom3'].append(forces_mag3)
        forces_mag_dict['atom15'].append(forces_mag15)
        forces_mag_dict['atom16'].append(forces_mag16)

        positions_list.append(atom.get_positions())

    forces_mag_atom3 = np.concatenate(forces_mag_dict['atom3'], axis=0)
    forces_mag_atom15 = np.concatenate(forces_mag_dict['atom15'], axis=0)
    forces_mag_atom16 = np.concatenate(forces_mag_dict['atom16'], axis=0)

    return forces_mag_atom3, forces_mag_atom15, forces_mag_atom16, np.array(positions_list)

# %%
# Get force magnitude distribution
trgj_dft = read('/scratch/yangzd/materials/data/md/lips/lips.xyz', index=':', format='extxyz') # Replace it with your own path
forces_mag_Li_dft, forces_mag_P_dft, forces_mag_S_dft, positions_dft = force_mag_per_atom(trgj_dft) 
md_dir = Path('./wandb/run-20240623_084150-E2GNN_LiPS_20240623_084148/md') # Replace it with your own path
traj_ml = Trajectory(md_dir / 'atoms.traj')
forces_mag_Li_ml, forces_mag_P_ml, forces_mag_S_ml, positions_ml = force_mag_per_atom(traj_ml)

# Define colors for each atom type
colors = {'Li': '#1f77b4', 'P': '#ff7f0e', 'S': '#2ca02c'}  # colors from the "tab10" palette

# Define labels and line styles for each method
method_specs = {'DFT': {'label': 'DFT', 'linestyle': '-'}, 'ML': {'label': 'EIGNN', 'linestyle': (0, (1, 1))}}

# Define bins for histogram
bins = np.linspace(0, max(np.max(forces_mag_Li_dft), np.max(forces_mag_P_dft), np.max(forces_mag_S_dft)), 60)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Calculate and plot histograms
for method, forces in zip(['DFT', 'ML'], [[forces_mag_Li_dft, forces_mag_P_dft, forces_mag_S_dft],
                                          [forces_mag_Li_ml, forces_mag_P_ml, forces_mag_S_ml]]):
    for atom_type, force in zip(['Li', 'P', 'S'], forces):
        hist, bin_edges = np.histogram(force, bins=bins, density=True)
        ax.step(bin_edges[:-1], hist, where='post', color=colors[atom_type], 
                label=f'{method_specs[method]["label"]} {atom_type}', linestyle=method_specs[method]['linestyle'], lw=1.5)

ax.set_xlabel(r'$\vert \mathbf{\vec{F}} \vert$ ($\rm eV \AA^{-1}$)', fontsize=20)
ax.set_ylabel(r'PDF($\vert \mathbf{\vec{F}} \vert$)', fontsize=20)
ax.legend(fontsize=18, loc='upper right')
ax.set_xlim(0, 4)  # Set the x-axis limits
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(md_dir, "force_pdf.jpg"), dpi=400)

positions_dft = torch.from_numpy(positions_dft)
positions_ml = torch.from_numpy(positions_ml)

# %%
# get hr
xlim = 10
n_bins = 500
bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing h(r)

print(positions_dft[::5].shape)
print(positions_ml.shape)

gt_hist = get_hr(positions_dft[::5], bins)
pred_hist = get_hr(positions_ml, bins)
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(bins[2:], gt_hist[1:], label='Reference', linewidth=3, linestyle='-', c='#1f77b4')
plt.plot(bins[2:], pred_hist[1:], label='Prediction', linewidth=3, linestyle='--', c='#ff7f0e')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel(r'r ($\AA$)', fontsize=20)
ax.set_ylabel(r'h(r) ($\AA^{-1}$)', fontsize=20)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.legend(fontsize=20, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(md_dir, "hr.jpg"), dpi=400)

hr = mae(gt_hist[1:], pred_hist[1:], xlim)
print(hr)

# %%
stability_threshold = 1.0
rdf_check_interval = 80 # 1 ps. recording freq is 0.0125 ps. 

xlim = 6
nbins = 500
bins = np.linspace(1e-6, xlim, nbins + 1)

trgj_dft = read('/scratch/yangzd/materials/data/md/lips/lips.xyz', index=':', format='extxyz') # Replace it with your own path
n_points = len(trgj_dft)
positions, cell, atomic_numbers = [], [], []
for i in range(n_points):
    positions.append(trgj_dft[i].get_positions())
    cell.append(trgj_dft[i].get_cell())
    atomic_numbers.append(trgj_dft[i].get_atomic_numbers())
positions = torch.from_numpy(np.array(positions))
cell = torch.from_numpy(np.array(cell)[0])
atomic_numbers = torch.from_numpy(np.array(atomic_numbers)[0])

# unwrap positions
all_displacements = []
for i in (range(1, len(positions))):
    next_pos = unwrap(positions[i-1], positions[i], cell)
    displacements = next_pos - positions[i-1]
    all_displacements.append(displacements)
displacements = torch.stack(all_displacements)
accum_displacements = torch.cumsum(displacements, dim=0)
positions = torch.cat([positions[0].unsqueeze(0), positions[0] + accum_displacements], dim=0)

# gt_rdf = np.load(md_dir, "gt_rdf.npy")
gt_rdf = get_lips_rdf(positions[::], cell, bins, device='cpu')
np.save(os.path.join(md_dir, "gt_rdf.npy"), gt_rdf)
# Li diffusivity unit in m^2/s. remove the first 5 ps as equilibrium.
# Desirably, we want longer trajectories for computing diffusivity.
gt_diff = get_smoothed_diff((positions[2500:None:25, atomic_numbers == 3])) * 20 * 1e-8

run = load_run(md_dir, atomic_numbers, cell, xlim, bins, stability_threshold, gt_rdf, gt_diff)

write_pickle(os.path.join(md_dir, "run.pkl"), run)

xaxis = np.linspace(1e-6, xlim, nbins)
plt.plot(xaxis, gt_rdf, label='Reference', linewidth=2, linestyle='-')
plt.plot(xaxis, run['rdf'], label='Prediction', linewidth=2, linestyle='--')
plt.legend()

# %%
collapse_ps = (run['collapse_pt']-1) / 80
rdf_mae = run['rdf_error']
diff_mae = run['diff_error'] * 1e9
print(f'stability: {collapse_ps:.1f} ps \nRDF mae: {rdf_mae:.2f}' +
      f'\nDiffusivity mae: {diff_mae:.2f} x 10^-9 m^2/s')

# %%
xlim = 6
nbins = 500
run = read_pickle(os.path.join(md_dir, "run.pkl"))
gt_rdf = np.load(os.path.join(md_dir, "gt_rdf.npy"))
xaxis = np.linspace(1e-6, xlim, nbins)

# fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(xaxis, gt_rdf, label='Reference', linewidth=3, linestyle='-', c='#1f77b4')
plt.plot(xaxis, run['rdf'], label='Prediction', linewidth=3, linestyle='--', c='#ff7f0e')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel(r'r ($\AA$)', fontsize=20)
ax.set_ylabel(r'RDF(r)', fontsize=20)
plt.tight_layout()
plt.legend(fontsize=20, loc='upper right')
plt.savefig(os.path.join(md_dir, "rdf.jpg"), dpi=400)

# %%
