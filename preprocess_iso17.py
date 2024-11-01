# %%
import os
from ase.db import connect
import multiprocessing as mp
import argparse
import lmdb
import numpy as np
import torch
from tqdm import tqdm
from graph_constructor import AtomsToGraphs
import pandas as pd
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

def write_data(mp_args):
    a2g, atom_db_path, db_path, data_indices = mp_args
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    with connect(atom_db_path) as conn:
        for i, index in enumerate(tqdm(data_indices, desc='Reading atoms objects', position=0, leave=True)):
            row = conn.get(int(index))
            atoms = row.toatoms()
            energy = row['total_energy']
            forces = row.data['atomic_forces']

            # Process the atoms object as needed
            data = a2g.convert(atoms)
            data.fid = index
            data.y = energy
            forces = torch.Tensor(forces)
            data.force = forces

            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i+1, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    
    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers

    train_indices_path = os.path.join(data_root, 'train_ids.txt')
    val_indices_path = os.path.join(data_root, 'validation_ids.txt')

    reference_db_path = os.path.join(data_root, 'reference.db')
    test_within_db_path = os.path.join(data_root, 'test_within.db')
    test_other_db_path = os.path.join(data_root, 'test_other.db')

    train_indices = np.squeeze(pd.read_csv(train_indices_path, header=None).to_numpy())
    val_indices = np.squeeze(pd.read_csv(val_indices_path, header=None).to_numpy())

    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=5,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
        r_edges=False,
    )

    # process the two test sets.
    for dataset in ['test_within', 'test_other']:
        if dataset == 'test_within':
            atom_db_path = test_within_db_path
            db_path = atom_db_path.replace('test_within.db', 'test_within_dataset')
        else:
            atom_db_path = test_other_db_path
            db_path = atom_db_path.replace('test_other.db', 'test_other_dataset')

        data_len = None
        with connect(atom_db_path) as conn:
            data_len = len(conn)

        print(f'{dataset}: {data_len}')

        data_indices = np.array(list(range(1, data_len+1)))
        save_path = Path(db_path)
        save_path.mkdir(parents=True, exist_ok=True)

        mp_db_paths = [
            os.path.join(save_path, "data.%04d.lmdb" % i)
            for i in range(num_workers)
        ]
        mp_data_indices = np.array_split(data_indices, num_workers)

        pool = mp.Pool(num_workers)
        mp_args = [
            (
                a2g,
                atom_db_path,
                mp_db_paths[i],
                mp_data_indices[i]
            )
            for i in range(num_workers)
        ]

        pool.imap(write_data, mp_args)

        pool.close()
        pool.join()

    # process the training and validation sets.
    for dataset in ['train', 'val']:
        atom_db_path = reference_db_path

        if dataset == 'train':
            data_indices = train_indices
            db_path = atom_db_path.replace('reference.db', 'train_dataset')
        else:
            data_indices = val_indices
            db_path = atom_db_path.replace('reference.db', 'valid_dataset')

        save_path = Path(db_path)
        save_path.mkdir(parents=True, exist_ok=True)

        mp_db_paths = [
            os.path.join(save_path, "data.%04d.lmdb" % i)
            for i in range(num_workers)
        ]
        mp_data_indices = np.array_split(data_indices, num_workers)

        pool = mp.Pool(num_workers)
        mp_args = [
            (
                a2g,
                atom_db_path,
                mp_db_paths[i],
                mp_data_indices[i]
            )
            for i in range(num_workers)
        ]

        pool.imap(write_data, mp_args)

        pool.close()
        pool.join()