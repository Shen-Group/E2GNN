# %%
import os
from ase.io.vasp import read_vasp_xml
import multiprocessing as mp
import os
import lmdb
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import pickle
from graph_constructor import AtomsToGraphs
from sklearn.model_selection import train_test_split
import argparse
from lmdb_dataset import TrajectoryLmdbDataset
import warnings
warnings.filterwarnings("ignore")

# %%
def save_metadata(data_root):
    train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train')})
   
    energy_list = []
    force_list = []
    for data in train_set:
        energy_list.append(data[0].y)
        force_list.append(data[0].force)

    energy = torch.cat(energy_list).numpy()
    force = torch.stack(force_list).numpy()

    norm_stats = {
        'e_mean': energy.mean(),
        'e_std': energy.std(),
        'f_mean': force.mean(),
        'f_std': force.std(),
    }
    save_path = Path(data_root)
    np.save(save_path / 'metadata', norm_stats)
    path = save_path / 'metadata.npy'
    print("norm_stats: ", np.load(path, allow_pickle=True).item())

def write_data(mp_args):
    a2g, db_path, atoms_list, atoms_indices = mp_args

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for i, atoms_index in enumerate(tqdm(atoms_indices, desc='Reading atoms objects', position=0, leave=True)):
        atoms = atoms_list[atoms_index]
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        data = a2g.convert(atoms)
        data.y = torch.Tensor([energy])
        data.force = torch.Tensor(forces)

        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i+1, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

# get_metadata()
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')

    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers

    data_path = os.path.join(data_root, 'vasprun.xml') 
    configs = read_vasp_xml(data_path, index=slice(None)) # Read xml

    atoms_list = []
    for i, atoms in enumerate(configs):
        atoms_list.append(atoms)

    data_indices = np.array(list(range(len(atoms_list))))
    train_indices, test_indices = train_test_split(data_indices, test_size=0.2, train_size=0.8, random_state=123, shuffle=True)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.1, train_size=0.9, random_state=123, shuffle=True)
    
    print("train: ", len(train_indices))
    print("val: ", len(val_indices))
    print("test: ", len(test_indices))

    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=5,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=False,
    )

    for dataset in ['train', 'val', 'test']:
        db_path = os.path.join(data_root, dataset)
        save_path = Path(db_path)
        save_path.mkdir(parents=True, exist_ok=True)
        mp_db_paths = [
            os.path.join(save_path, "data.%04d.lmdb" % i)
            for i in range(num_workers)
        ]
        if dataset == 'train':
            mp_data_indices = np.array_split(train_indices, num_workers)
        elif dataset == 'val':
            mp_data_indices = np.array_split(val_indices, num_workers)
        elif dataset == 'test':
            mp_data_indices = np.array_split(test_indices, num_workers)

        pool = mp.Pool(num_workers)
        mp_args = [
            (
                a2g,
                mp_db_paths[i],
                atoms_list,
                mp_data_indices[i]
            )
            for i in range(num_workers)
        ]

        pool.imap(write_data, mp_args)
        pool.close()
        pool.join()

    save_metadata(data_root)

# %%