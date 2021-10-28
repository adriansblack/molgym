import torch
from torch_geometric.data import Batch

from .torch_tools import TensorDict


def neg_log_likelihood(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # logp: [n_graphs, ]
    return -1 * torch.mean(pred['logp'])  # []
