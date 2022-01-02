import copy
from typing import Tuple

import torch
from e3nn import o3

from .policy import Policy
from .q_function import QFunction


def requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


class SACAgent(torch.nn.Module):
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
        beta: float,
    ):
        super().__init__()

        self.policy = Policy(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            network_width=network_width,
            num_gaussians=num_gaussians,
            min_max_distance=min_max_distance,
            beta=beta,
        )

        self.q1, self.q2 = tuple(
            QFunction(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                max_ell=max_ell,
                num_interactions=num_interactions,
                num_elements=num_elements,
                hidden_irreps=hidden_irreps,
                network_width=network_width,
            ) for _ in range(2))

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)


class SACTarget(torch.nn.Module):
    def __init__(self, agent: SACAgent):
        super().__init__()
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/2
        self.q1 = copy.deepcopy(agent.q1)
        self.q2 = copy.deepcopy(agent.q2)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
