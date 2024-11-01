# %%
import math
import torch
from torch import nn
from torch_scatter import scatter
from graph_utils import ScaledSiLU, AtomEmbedding, RadialBasis, radius_graph_pbc
from torch_geometric.nn import radius_graph
from torch_geometric.nn import global_mean_pool


class GlobalScalar(nn.Module):
    def __init__(self, in_feats, out_feats, residual=False):
        super(GlobalScalar, self).__init__()
        # Add residual connection or not
        self.residual = residual

        self.mlp_vn = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU())
        
        self.mlp_node = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU())

    def update_local_emb(self, x, batch, vx):
        h = self.mlp_node(torch.cat([x, vx[batch]], dim=1)) + x

        return h, vx

    def update_global_emb(self, x, batch, vx):
        vx_temp = self.mlp_vn(torch.cat([global_mean_pool(x, batch), vx], dim=-1)) 

        if self.residual:
            vx = vx + vx_temp
        else:
            vx = vx_temp

        return vx

class GlobalVector(nn.Module):
    def __init__(self, in_feats, out_feats, residual=False):
        super(GlobalVector, self).__init__()
        # Add residual connection or not
        self.residual = residual

        self.mlp_vn = nn.Linear(in_feats, out_feats, bias=False)
        self.mlp_node = nn.Linear(in_feats, out_feats, bias=False)
 
    def update_local_emb(self, vec, batch, vvec):
        hvec = self.mlp_node(vec + vvec[batch]) + vec

        return hvec, vvec

    def update_global_emb(self, vec, batch, vvec):
        vvec_temp = self.mlp_vn(scatter(vec, batch, dim=0, reduce='mean', dim_size=vvec.size(0)) + vvec) 

        if self.residual:
            vvec = vvec + vvec_temp
        else:
            vvec = vvec_temp

        return vvec

class E2GNNMessage(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ):
        super(E2GNNMessage, self).__init__()

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels*3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels*3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        j, i = edge_index

        rbf_h = self.rbf_proj(edge_rbf)

        x_h = self.x_proj(x)
        x_ji1, x_ji2, x_ji3 = torch.split(x_h[j] * rbf_h * self.inv_sqrt_3, self.hidden_channels, dim=-1)

        vec_ji = x_ji1.unsqueeze(1) * vec[j] + x_ji2.unsqueeze(1) * edge_vector.unsqueeze(2)
        vec_ji = vec_ji * self.inv_sqrt_h

        d_vec = scatter(vec_ji, index=i, dim=0, dim_size=x.size(0)) 
        d_x = scatter(x_ji3, index=i, dim=0, dim_size=x.size(0)) 
        
        return d_x, d_vec
    
class E2GNNUpdate(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels*2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels*3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )

        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        gate = torch.tanh(xvec3)
        dx = xvec2 * self.inv_sqrt_2 + x * gate
        dvec = xvec1.unsqueeze(1) * vec1

        return dx, dvec
    
class E2GNN(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.0,
        max_neighbors=20,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        regress_forces=True,
        direct_forces=True,
        use_pbc=False,
        otf_graph=True,
        num_elements=83,
    ):
        super(E2GNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.vn_emb = nn.Embedding(1, hidden_channels)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        self.global_vector_layers = nn.ModuleList()
        self.global_scalar_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_layers.append(
                E2GNNMessage(hidden_channels, num_rbf)
            )
            self.update_layers.append(E2GNNUpdate(hidden_channels))

            self.global_vector_layers.append(GlobalVector(hidden_channels, hidden_channels, residual=True))
            self.global_scalar_layers.append(GlobalScalar(hidden_channels, hidden_channels, residual=True))

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.out_forces = nn.Linear(hidden_channels, 1, bias=False)
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        natoms = data.natoms
        z = data.atomic_numbers.long()

        assert z.dim() == 1 and z.dtype == torch.long

        if self.otf_graph:
            if self.use_pbc:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, self.cutoff, self.max_neighbors
                )
                cell = data.cell
                j, i = edge_index
                cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).float()
                abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)
                vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]
                edge_dist = vecs.norm(dim=-1)
                edge_vector = -vecs/edge_dist.unsqueeze(-1)
            else:
                edge_index = radius_graph(pos, self.cutoff, batch, max_num_neighbors=self.max_neighbors)
                j, i = edge_index
                vecs = pos[j] - pos[i]
                edge_dist = vecs.norm(dim=-1)
                edge_vector = -vecs/edge_dist.unsqueeze(-1)
        else:
            if self.use_pbc:
                edge_index, cell, cell_offsets, neighbors = data.edge_index, data.cell, data.cell_offsets, data.neighbors
                abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)
                j, i = edge_index
                cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).float()
                vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]
                edge_dist = vecs.norm(dim=-1)
                edge_vector = -vecs/edge_dist.unsqueeze(-1)
            else:
                edge_index = data.edge_index
                j, i = edge_index
                vecs = pos[j] - pos[i]
                edge_dist = vecs.norm(dim=-1)
                edge_vector = -vecs/edge_dist.unsqueeze(-1)       

        edge_rbf = self.radial_basis(edge_dist)  # rbf * evelope
        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        vx = self.vn_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        vvec = torch.zeros(
                batch[-1].item() + 1, 3, x.size(1)).to(edge_index.dtype).to(edge_index.device)
        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            x, vx = self.global_scalar_layers[i].update_local_emb(x, batch, vx)
            vec, vvec = self.global_vector_layers[i].update_local_emb(vec, batch, vvec)

            dx, dvec = self.message_layers[i](
                x, vec, edge_index, edge_rbf, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)
            x = x + dx
            vec = vec + dvec

            vx = self.global_scalar_layers[i].update_global_emb(x, batch, vx)
            vvec = self.global_vector_layers[i].update_global_emb(vec, batch, vvec)

        #### Output block #####################################################

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0)

        forces = self.out_forces(vec).squeeze(-1)
        return energy, forces

# %%