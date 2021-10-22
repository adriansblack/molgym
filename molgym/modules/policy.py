import torch.nn
from e3nn import o3

from molgym.graph_categorical import GraphCategoricalDistribution
from .blocks import MLP
from .models import SimpleModel
from ..data.geometric_data import StateActionBatch


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
    ):
        super().__init__()

        # Embedding
        self.embedding = SimpleModel(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
        )
        self.cov_irreps = self.embedding.irreps_out

        # Norm
        self.norm = o3.Norm(irreps_in=self.cov_irreps)
        self.inv_dim = self.norm.irreps_out.dim

        self.bag_tp = o3.FullyConnectedTensorProduct(self.cov_irreps, o3.Irreps(f'{num_elements}x0e'), self.cov_irreps)

        self.phi_focus = MLP(
            input_dim=self.inv_dim,
            output_dims=(network_width, 1),
            gate=torch.nn.ReLU(),
        )

    def forward(self, data: StateActionBatch):
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

        focused_inv = s_inv[focus + data.ptr[:-1]]

        return s_cov, s_inv, focus_logits
