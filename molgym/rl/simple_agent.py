import copy
from typing import Optional, Dict, Union, Any, Tuple

import numpy as np
import torch
import torch_scatter
from e3nn import o3

from molgym.data import StateBatch, FOCUS_KEY, ELEMENT_KEY, DISTANCE_KEY, ORIENTATION_KEY
from molgym.distributions import GraphCategoricalDistribution
from molgym.tools import TensorDict, masked_softmax
from .q_function import QFunction


class SimplePolicy(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        min_max_distance: Tuple[float, float],
    ):
        super().__init__()
        self.num_elements = num_elements

        # Distance
        min_distance, max_distance = min_max_distance
        self.d_center = torch.tensor((min_distance + max_distance) / 2)
        self.d_half_width = torch.tensor((max_distance - min_distance) / 2)

        self.d_mean_trans = torch.nn.Parameter(torch.tensor([[0.0]]), requires_grad=True)
        self.d_log_stds = torch.nn.Parameter(torch.tensor([np.log(0.2)]), requires_grad=True)

        self.orientation_template = torch.tensor([[1.0, 0.0, 0.0]])

    def forward(
        self,
        state: StateBatch,
        action: Optional[TensorDict] = None,
        training=False,
    ) -> Tuple[Dict[str, Union[torch.Tensor, TensorDict]], Dict[str, Any]]:

        focus_logits = torch.ones(size=(state.elements.shape[0], ), device=state.elements.device)  # Uniform
        focus_probs = torch_scatter.scatter_softmax(src=focus_logits, index=state.batch, dim=-1)  # [num_nodes,]
        focus_distr = GraphCategoricalDistribution(probs=focus_probs, batch=state.batch, ptr=state.ptr)

        # Focus
        if action is not None:
            focus = action[FOCUS_KEY]
        elif training:
            focus = focus_distr.sample()
        else:
            focus = focus_distr.argmax()

        # Element
        all_element_logits = torch.ones(size=(state.elements.shape[0], self.num_elements),
                                        device=state.elements.device)  # Uniform
        all_element_probs = masked_softmax(all_element_logits, mask=(state.bag > 0)[state.batch])  # [n_nodes, n_z]
        focused_element_probs = all_element_probs[focus + state.ptr[:-1]]  # [n_graphs, n_z]
        element_distr = torch.distributions.Categorical(probs=focused_element_probs)

        if action is not None:
            element = action[ELEMENT_KEY]
        elif training:
            element = element_distr.sample()
        else:
            element = torch.argmax(element_distr.probs, dim=-1)

        # Distance
        d_mean = torch.tanh(self.d_mean_trans.expand(state.num_graphs, -1)) * self.d_half_width + self.d_center
        d_distr = torch.distributions.Normal(loc=d_mean, scale=torch.exp(self.d_log_stds).clamp(min=1e-6))

        if action is not None:
            distance = action[DISTANCE_KEY]
        elif training:
            distance = d_distr.sample()
        else:
            distance = d_mean

        # Orientation
        if action is not None:
            orientation = action[ORIENTATION_KEY]
        else:
            orientation = self.orientation_template.repeat(state.num_graphs, 1)

        # Log probs
        log_prob_list = [
            d_distr.log_prob(distance),
        ]
        log_prob = torch.stack(log_prob_list, dim=-1).sum(dim=-1)  # [n_graphs, ]

        response: Dict[str, Union[torch.Tensor, TensorDict]] = {
            'action': {
                FOCUS_KEY: focus,
                ELEMENT_KEY: element,
                DISTANCE_KEY: distance,
                ORIENTATION_KEY: orientation,
            },
            'logp': log_prob,
        }
        aux = {
            'distrs': [d_distr],
        }

        return response, aux


def requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


class SimpleSACAgent(torch.nn.Module):
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
        min_max_distance: Tuple[float, float],
    ):
        super().__init__()

        self.policy = SimplePolicy(
            num_elements=num_elements,
            min_max_distance=min_max_distance,
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

    def freeze_q(self):
        requires_grad(self.q1, False)
        requires_grad(self.q2, False)

    def unfreeze_q(self):
        requires_grad(self.q1, True)
        requires_grad(self.q2, True)


class SimpleSACTarget(torch.nn.Module):
    def __init__(self, agent: SimpleSACAgent):
        super().__init__()
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/2
        self.q1 = copy.deepcopy(agent.q1)
        self.q2 = copy.deepcopy(agent.q2)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
