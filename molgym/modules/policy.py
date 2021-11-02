from typing import Tuple, Any, Dict

import numpy as np
import torch.nn
import torch_scatter
from e3nn import o3

from molgym.data import StateActionBatch
from molgym.distributions import (GaussianMixtureModel, GraphCategoricalDistribution, SO3Distribution,
                                  compute_ef_cond_entropy)
from molgym.tools import masked_softmax, to_one_hot, TensorDict
from .blocks import MLP
from .irreps_tools import get_merge_instructions
from .models import SimpleModel
from .radial import BesselBasis


class Policy(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        network_width: int,
        num_gaussians: int,
        min_max_distance: Tuple[float, float],
        gamma: float,
    ):
        super().__init__()

        self.num_elements = num_elements
        self.ell_max = max_ell
        self.gamma = gamma

        # Embedding
        self.embedding = SimpleModel(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            num_interactions=num_interactions,
            num_elements=self.num_elements,
            hidden_irreps=hidden_irreps,
        )
        self.cov_irreps = self.embedding.irreps_out

        # Norm
        self.norm = o3.Norm(irreps_in=self.cov_irreps)
        self.inv_dim = self.norm.irreps_out.dim

        z_irreps = o3.Irreps(f'{self.num_elements}x0e')
        self.bag_tp = o3.FullyConnectedTensorProduct(self.cov_irreps, z_irreps, self.cov_irreps)

        # Focus
        self.phi_focus = MLP(
            input_dim=self.inv_dim,
            output_dims=(network_width, 1),
            gate=torch.nn.ReLU(),
        )

        # Element
        self.phi_element = MLP(
            input_dim=self.inv_dim,
            output_dims=(network_width, num_elements),
            gate=torch.nn.ReLU(),
        )

        # Distance
        self.num_gaussians = num_gaussians
        self.phi_distance = MLP(
            input_dim=self.inv_dim + num_elements,
            output_dims=(network_width, 2 * self.num_gaussians),
            gate=torch.nn.ReLU(),
        )

        self.min_max_distance = min_max_distance
        min_distance, max_distance = self.min_max_distance
        self.d_center = torch.tensor((min_distance + max_distance) / 2)
        self.d_half_width = torch.tensor((max_distance - min_distance) / 2)
        self.d_log_stds = torch.nn.Parameter(torch.tensor([np.log(0.1)] * self.num_gaussians), requires_grad=True)

        # Orientation
        self.bessel_fn = BesselBasis(r_max=max_distance, num_basis=num_bessel)
        sph_irreps = o3.Irreps.spherical_harmonics(self.ell_max)
        instructions = get_merge_instructions(self.cov_irreps, z_irreps, sph_irreps)
        self.mix_tp = o3.TensorProduct(self.cov_irreps,
                                       z_irreps,
                                       irreps_out=sph_irreps,
                                       instructions=instructions,
                                       shared_weights=False,
                                       internal_weights=False)
        self.mix_tp_weights = o3.Linear(o3.Irreps(f'{num_bessel}x0e'), o3.Irreps(f'{self.mix_tp.weight_numel}x0e'))

    def forward(self, data: StateActionBatch) -> Tuple[TensorDict, Dict[str, Any]]:
        s_inter = self.embedding(data)
        s_cov = self.bag_tp(s_inter, data.bag[data.batch])
        s_inv = self.norm(s_cov)

        focus_logits = self.phi_focus(s_inv).squeeze(-1)  # [n_nodes, ]
        focus_probs = torch_scatter.scatter_softmax(src=focus_logits, index=data.batch, dim=-1)  # [num_nodes,]
        focus_distr = GraphCategoricalDistribution(probs=focus_probs, batch=data.batch, ptr=data.ptr)

        # Focus
        if hasattr(data, 'focus') and data.focus is not None:
            focus = data.focus
        else:
            focus = focus_distr.sample()

        # Element
        all_element_logits = self.phi_element(s_inv)  # [n_nodes, n_z]
        all_element_probs = masked_softmax(all_element_logits, mask=(data.bag > 0)[data.batch])  # [n_nodes, n_z]
        focused_element_probs = all_element_probs[focus + data.ptr[:-1]]  # [n_graphs, n_z]
        element_distr = torch.distributions.Categorical(probs=focused_element_probs)

        if hasattr(data, 'element') and data.element is not None:
            element = data.element
        else:
            element = element_distr.sample()

        element_oh = to_one_hot(element.unsqueeze(-1), num_classes=self.num_elements)

        # Distance
        focused_inv = s_inv[focus + data.ptr[:-1]]
        d_input = torch.cat([focused_inv, element_oh], dim=-1)  # [n_graphs, n_hidden + n_z]
        gmm_log_probs, d_mean_trans = self.phi_distance(d_input).split(self.num_gaussians, dim=-1)
        d_mean = torch.tanh(d_mean_trans) * self.d_half_width + self.d_center
        d_distr = GaussianMixtureModel(log_probs=gmm_log_probs,
                                       means=d_mean,
                                       stds=torch.exp(self.d_log_stds).clamp(min=1e-6))

        if hasattr(data, 'distance') and data.distance is not None:
            distance = data.distance
        else:
            distance = d_distr.sample()

        # Orientation
        focused_cov = s_cov[focus + data.ptr[:-1]]
        tp_weights = self.mix_tp_weights(self.bessel_fn(distance.unsqueeze(-1)))
        cond_cov = self.mix_tp(focused_cov, element_oh, tp_weights)  # [n_graphs, irreps]
        spherical_distr = SO3Distribution(cond_cov, lmax=self.ell_max, gamma=self.gamma)

        if hasattr(data, 'orientation') and data.orientation is not None:
            orientation = data.orientation
        else:
            orientation = spherical_distr.sample()

        # Log probs
        log_prob_list = [
            focus_distr.log_prob(focus),
            element_distr.log_prob(element),
            d_distr.log_prob(distance),
            spherical_distr.log_prob(orientation),
        ]
        log_prob = torch.stack(log_prob_list, dim=-1).sum(dim=-1)  # [n_graphs, ]

        # Entropy
        entropy_list = [
            focus_distr.entropy(),
            compute_ef_cond_entropy(focus_probs=focus_probs,
                                    element_probs=all_element_probs,
                                    batch=data.batch,
                                    num_graphs=data.num_graphs),
        ]
        entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)  # [n_graphs, ]

        response = {
            'focus': focus,
            'element': element,
            'distance': distance,
            'orientation': orientation,
            'logp': log_prob,
            'entropy': entropy,
        }
        aux = {
            'distrs': [focus_distr, element_distr, d_distr, spherical_distr],
        }

        return response, aux
