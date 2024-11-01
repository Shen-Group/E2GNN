
# %%
if __name__ =='__main__':
    import os
    import torch
    from torch_geometric.data import Batch
    from glob import glob
    from graph_constructor import AtomsToGraphs
    from ase.io import read
    from ase.build import make_supercell
    from E2GNN import E2GNN

    import warnings
    warnings.filterwarnings("ignore")

    data_root = './extensivity'
    data_paths = glob(os.path.join(data_root, "*"))

    for data_path in data_paths:
        atoms = read(data_path)

        # Define the transformation matrix for a 2x1x1 supercell
        transformation_matrix = [[2, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]

        # Create the supercell
        supercell_atoms = make_supercell(atoms, transformation_matrix)

        a2g = AtomsToGraphs(
            max_neigh=50,
            radius=5,
        )
        graph_single = a2g.convert(atoms)
        graph_double = a2g.convert(supercell_atoms)

        data_list = [graph_single, graph_double]

        batch = Batch.from_data_list(data_list)
        n_neighbors = []
        for i, data in enumerate(data_list):
            n_index = data.edge_index[1, :]
            n_neighbors.append(n_index.shape[0])
        batch.neighbors = torch.tensor(n_neighbors)

        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=5.0, max_neighbors=50, use_pbc=True, otf_graph=False, num_elements=118)

        with torch.no_grad():
            energy, _ = model(batch)

            print("Single: ", energy[0])
            print("Double: ", energy[1])
            print("-" * 20)

# %%