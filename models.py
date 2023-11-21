import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class CustomGAT(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        self.weight_matrix = nn.Linear(input_features, output_features)
        self.attention_matrix = nn.Parameter(torch.zeros(size=(output_features, output_features)))
        self.gating_mechanism = nn.Linear(input_features + output_features, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, node_input: Tensor, adjacency_matrix: Tensor) -> Tensor:
        node_transform = self.weight_matrix(node_input)
        edge_attention = torch.einsum("ijl,ikl->ijk", (torch.matmul(node_transform, self.attention_matrix), node_transform))
        edge_attention = edge_attention + edge_attention.permute((0, 2, 1))

        zero_vector = -9e15 * torch.ones_like(edge_attention)
        attention_weights = torch.where(adjacency_matrix > 1e-6, edge_attention, zero_vector)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights * adjacency_matrix
        node_output = F.relu(torch.einsum("aij,ajk->aik", (attention_weights, node_transform)))

        gating_coefficient = torch.sigmoid(self.gating_mechanism(torch.cat([node_input, node_output], -1))).repeat(
            1, 1, node_input.size(-1)
        )
        updated_nodes = gating_coefficient * node_input + (1 - gating_coefficient) * node_output
        return updated_nodes

class InteractionNetwork(nn.Module):
    def __init__(self, atom_features: int):
        super().__init__()

        self.weight_transform = nn.Linear(atom_features, atom_features)
        self.message_transform = nn.Linear(atom_features, atom_features)
        self.state_update = nn.GRUCell(atom_features, atom_features)

    def forward(self, atom_input_1: Tensor, atom_input_2: Tensor, edge_validity: Tensor) -> Tensor:
        extended_edges = atom_input_2.unsqueeze(1).repeat(1, atom_input_1.size(1), 1, 1)

        transformed_1 = self.weight_transform(atom_input_1)
        transformed_2 = (self.message_transform(extended_edges) * edge_validity.unsqueeze(-1)).max(2)[0]
        concatenated_features = F.relu(transformed_1 + transformed_2)
        feature_dimensions = concatenated_features.size(-1)
        updated_features = self.state_update(concatenated_features.reshape(-1, feature_dimensions), atom_input_1.reshape(-1, feature_dimensions))
        updated_features = updated_features.reshape(atom_input_1.size(0), atom_input_1.size(1), atom_input_1.size(2))
        return updated_features

class ConvolutionBlock(nn.Module):
    def __init__(self, input_feat: int, output_feat: int, dropout: float = 0.0, stride: int = 1, kernel_size: int = 3, padding: int = 1, use_bn: bool = True):
        super().__init__()

        layers = []
        layers.append(nn.Conv3d(input_feat, output_feat, kernel_size, stride, padding))
        if use_bn:
            layers.append(nn.BatchNorm3d(output_feat))
        layers.append(nn.ReLU())
        if dropout != 0:
            layers.append(nn.Dropout3d(p=dropout))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.conv_block(input_tensor)

class PredictionBlock(nn.Module):
    def __init__(self, input_feat: int, output_feat: int, dropout_rate: float, is_final: bool):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_feat, output_feat))
        if not is_final:
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.ReLU())
        self.pred_block = nn.Sequential(*layers)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.pred_block(input_tensor)

class Model(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [CustomGAT(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.interaction_net:
            self.interaction_net = nn.ModuleList(
                [InteractionNetwork(args.dim_gnn) for _ in range(args.n_gnn)]
            )

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Tanh(),
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))

    def cal_hbond(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hbond_coeff * self.hbond_coeff)
        retval_val = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval, retval_val

    def cal_hydrophobic(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        retval_val = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval, retval_val

    def cal_vdw_interaction(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        ligand_valid: Tensor,
        target_valid: Tensor,) -> Tensor:
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(1, 1, target_valid.size(1))
        target_valid_ = target_valid.unsqueeze(1).repeat(1, ligand_valid.size(1), 1)
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )

        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm_0[dm_0 < 0.0001] = 1
        N = self.args.vdw_N
        vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.args.max_vdw_interaction - self.args.min_vdw_interaction)
        A = A + self.args.min_vdw_interaction

        energy = vdw_term1 + vdw_term2
        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_
        energy = A * energy
        # print("Shape of energies before sum in vDW: ", energy.shape)
        # print("Energies before sum in vDW: ", energy)
        energy_val = energy.sum(1).sum(1).unsqueeze(-1)
        return energy, energy_val

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float) -> Tensor:
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(
        self, sample: Dict[str, Any], DM_min: float = 0.5) -> Tuple[Tensor]:
        (
            ligand_h,
            ligand_adj,
            target_h,
            target_adj,
            interaction_indice,
            ligand_pos,
            target_pos,
            rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_valid,
            target_valid,
            ligand_non_metal,
            target_non_metal,
            _,
            _,) = sample.values()

        # feature embedding
        ligand_h = self.node_embedding(ligand_h)
        target_h = self.node_embedding(target_h)

        # distance matrix
        ligand_pos.requires_grad = True
        dm = self.cal_distance_matrix(ligand_pos, target_pos, DM_min)

        for idx in range(len(self.gconv)):
            ligand_h = self.gconv[idx](ligand_h, ligand_adj)
            target_h = self.gconv[idx](target_h, target_adj)
            ligand_h = F.dropout(
                ligand_h, training=self.training, p=self.args.dropout_rate
            )
            target_h = F.dropout(
                target_h, training=self.training, p=self.args.dropout_rate
            )

        # InteractionNet propagation
        if self.args.interaction_net:
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for idx in range(len(self.interaction_net)):
                new_ligand_h = self.interaction_net[idx](
                    ligand_h,
                    target_h,
                    adj12,
                )
                new_target_h = self.interaction_net[idx](
                    target_h,
                    ligand_h,
                    adj12.permute(0, 2, 1),
                )
                ligand_h, target_h = new_ligand_h, new_target_h
                ligand_h = F.dropout(
                    ligand_h, training=self.training, p=self.args.dropout_rate
                )
                target_h = F.dropout(
                    target_h, training=self.training, p=self.args.dropout_rate
                )

        # concat features
        h1_ = ligand_h.unsqueeze(2).repeat(1, 1, target_h.size(1), 1)
        h2_ = target_h.unsqueeze(1).repeat(1, ligand_h.size(1), 1, 1)
        h_cat = torch.cat([h1_, h2_], -1)

        # compute energy component
        energies = []
        
        # vdw interaction
        vdw_energy_mat, vdw_energy = self.cal_vdw_interaction(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
        )
        energies.append(vdw_energy)

        # hbond interaction
        hbond_energy_mat, hbond = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 0],
        )
        energies.append(hbond)

        # metal interaction
        metal_energy_mat, metal = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 1],
        )
        energies.append(metal)

        # hydrophobic interaction
        hydrophobic_energy_mat, hydrophobic = self.cal_hydrophobic(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 2],
        )
        energies.append(hydrophobic)

        energies = torch.cat(energies, -1)
        # rotor penalty
        if not self.args.no_rotor_penalty:
            energies = energies / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        gradient = torch.autograd.grad(
            energies.sum(), ligand_pos, retain_graph=True, create_graph=True
        )[0]
        der1 = torch.pow(gradient.sum(1), 2).mean()
        der2 = torch.autograd.grad(
            gradient.sum(), ligand_pos, retain_graph=True, create_graph=True
        )[0]
        der2 = -der2.sum(1).sum(1).mean()

        energies_mat = []
        energies_mat.append(vdw_energy_mat)
        energies_mat.append(hbond_energy_mat)
        energies_mat.append(metal_energy_mat)
        energies_mat.append(hydrophobic_energy_mat)
        # print("energies_mat length", len(energies_mat)) #DEBUG
        # print("energies_mat", energies_mat[0])
        return energies_mat, energies, der1, der2

