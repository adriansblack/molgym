import torch.nn
from e3nn import o3
from torch_scatter import scatter_sum

from molgym import modules, data


class QFunction(torch.nn.Module):
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
        infbag: bool = False,
    ) -> None:
        super().__init__()

        self.embedding = modules.SimpleModel(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
        )

        self.norm = o3.Norm(irreps_in=self.embedding.irreps_out)

        self.phi = modules.MLP(
            input_dim=self.norm.irreps_out.dim,
            output_dims=(network_width, network_width),
            gate=torch.nn.ReLU(),
        )

        if(infbag): num_elements = num_elements*2
        self.psi = modules.MLP(
            input_dim=network_width + num_elements,
            output_dims=(network_width, 1),
            gate=torch.nn.ReLU(),
        )

    def forward(
            self,
            next_state: data.StateBatch,  # next state, s_next
    ) -> torch.Tensor:  # [n_graphs, ]
        s_cov = self.embedding(next_state)  # [n_nodes, n_irrep]
        s_inv = self.norm(s_cov)  # invariant node feats [n_nodes, n_inv]
        node_prop = self.phi(s_inv)  # node propensity [n_nodes, n_width]
        graph_prop = scatter_sum(src=node_prop, index=next_state.batch, dim=0,
                                 dim_size=next_state.num_graphs)  # [n_graphs, n_width]

        # NOTE: for infinite bag setting, the cost of each atom and the total (remaining) volume will be needed
        return self.psi(torch.cat([graph_prop, next_state.bag], dim=-1)).squeeze(-1)  # [n_graphs, ]
