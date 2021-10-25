from typing import Tuple, Any, Dict

import numpy as np
import torch.nn
from e3nn import o3

from molgym.data import StateActionBatch
from molgym.gmm import GaussianMixtureModel
from molgym.graph_categorical import GraphCategoricalDistribution
from molgym.spherical_distrs import SO3Distribution
from molgym.tools import masked_softmax, to_one_hot
from .blocks import MLP
from .irreps_tools import tp_out_irreps_with_instructions
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
    ):
        super().__init__()

        self.num_elements = num_elements

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
        min_distance, max_distance = min_max_distance
        self.d_center = torch.tensor((min_distance + max_distance) / 2)
        self.d_half_width = torch.tensor((max_distance - min_distance) / 2)
        self.d_log_stds = torch.nn.Parameter(torch.tensor([np.log(0.1)] * self.num_gaussians), requires_grad=True)

        # Orientation
        self.bessel_fn = BesselBasis(r_max=max_distance, num_basis=num_bessel)
        irreps_mid, instructions = tp_out_irreps_with_instructions(self.cov_irreps, z_irreps, self.cov_irreps)
        self.mix_tp = o3.TensorProduct(self.cov_irreps,
                                       z_irreps,
                                       irreps_out=irreps_mid,
                                       instructions=instructions,
                                       shared_weights=False,
                                       internal_weights=False)
        self.mix_tp_weights = o3.Linear(o3.Irreps(f'{num_bessel}x0e'), o3.Irreps(f'{self.mix_tp.weight_numel}x0e'))

    def forward(self, data: StateActionBatch) -> Dict[str, Any]:
        s_inter = self.embedding(data)
        s_cov = self.bag_tp(s_inter, data.bag[data.batch])
        s_inv = self.norm(s_cov)

        focus_logits = self.phi_focus(s_inv).squeeze(-1)  # [n_nodes, ]
        focus_distr = GraphCategoricalDistribution(logits=focus_logits, batch=data.batch, ptr=data.ptr)

        # Focus
        if data.focus is not None:
            focus = data.focus
        else:
            focus = focus_distr.sample()

        focused_cov = s_cov[focus + data.ptr[:-1]]
        focused_inv = s_inv[focus + data.ptr[:-1]]

        # Element
        element_logits = self.phi_element(focused_inv)  # [n_graphs, n_zs]
        element_probs = masked_softmax(element_logits, mask=(data.bag > 0))  # [n_graphs, n_zs]
        element_distr = torch.distributions.Categorical(probs=element_probs)

        if data.element is not None:
            element = data.element
        else:
            element = element_distr.sample()

        element_oh = to_one_hot(element.unsqueeze(-1), num_classes=self.num_elements)

        # Distance
        d_input = torch.cat([focused_inv, element_oh], dim=-1)  # [n_graphs, n_hidden + n_zs]
        gmm_log_probs, d_mean_trans = self.phi_distance(d_input).split(self.num_gaussians, dim=-1)
        d_mean = torch.tanh(d_mean_trans) * self.d_half_width + self.d_center
        d_distr = GaussianMixtureModel(log_probs=gmm_log_probs,
                                       means=d_mean,
                                       stds=torch.exp(self.d_log_stds).clamp(min=1e-6))

        if data.distance is not None:
            distance = data.distance
        else:
            distance = d_distr.sample()

        # Orientation
        tp_weights = self.mix_tp_weights(self.bessel_fn(distance.unsqueeze(-1)))
        cond_cov = self.mix_tp(focused_cov, element_oh, tp_weights)  # [n_graphs, irreps]
        spherical_distr = SO3Distribution(cond_cov, lmax=3, gamma=1.0)

        # Log probs
        log_prob_list = [
            focus_distr.log_prob(focus),
            element_distr.log_prob(element),
            d_distr.log_prob(distance),
        ]
        log_prob = torch.stack(log_prob_list, dim=-1).sum(dim=-1)  # [n_graphs, ]

        return {
            'focus': focus,
            'element': element,
            'distance': distance,
            'logp': log_prob,
            'distrs': [focus_distr, element_distr, d_distr],
        }
