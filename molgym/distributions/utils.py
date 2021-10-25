import torch.nn
import torch_scatter
from torch.distributions.utils import probs_to_logits


def compute_ef_cond_entropy(
    focus_probs: torch.Tensor,  # [n_nodes, ]
    element_probs: torch.Tensor,  # [n_nodes, n_z]
    batch: torch.Tensor,  # [n_nodes, ]
    num_graphs: int,
):
    assert len(focus_probs.shape) == 1
    assert len(element_probs.shape) == 2
    assert focus_probs.shape[0] == element_probs.shape[0] == batch.shape[0]

    logp_pe_pf = probs_to_logits(element_probs) * element_probs * focus_probs.unsqueeze(-1)  # [n_nodes, n_z]
    sum_e = torch.sum(logp_pe_pf, dim=-1)  # [n_nodes, ]
    return -1 * torch_scatter.scatter_sum(sum_e, index=batch, dim=0, dim_size=num_graphs)
