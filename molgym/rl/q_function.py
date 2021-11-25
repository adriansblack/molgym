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

    def forward(
            self,
            batch_next: data.StateBatch,  # next state s'
    ) -> torch.Tensor:  # [n_graphs, ]
        s_cov = self.embedding(batch_next)  # [n_nodes, n_feats]
        s_inv = self.norm(s_cov)  # [n_nodes, n_inv]
        s_lat = torch.cat([s_inv, batch_next.bag[batch_next.batch]], dim=-1)

        # Local contribution to Q value
        q_node = self.phi(s_lat).squeeze(-1)  # [n_nodes, ]

        # Sum over all nodes in graph
        q_tot = scatter_sum(src=q_node, index=batch_next.batch, dim=-1, dim_size=batch_next.num_graphs)  # [n_graphs,]

        return q_tot
