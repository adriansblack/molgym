from abc import ABC

import torch


class GaussianMixtureModel(torch.distributions.MixtureSameFamily, ABC):
    def __init__(
        self,
        log_probs: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        validate_args=None,
    ) -> None:
        categoricals = torch.distributions.Categorical(logits=log_probs, validate_args=validate_args)
        normals = torch.distributions.Normal(loc=means, scale=stds, validate_args=validate_args)
        super().__init__(mixture_distribution=categoricals, component_distribution=normals, validate_args=validate_args)

    def argmax(self, count=128) -> torch.Tensor:
        # This can also be implemented using the EM algorithm
        # http://www.cs.columbia.edu/~jebara/htmlpapers/ARL/node61.html
        samples = self.sample(torch.Size((count, )))  # [S, B]
        log_probs = self.log_prob(samples)  # [S, B]
        indices = torch.argmax(log_probs, dim=0).unsqueeze(0)  # [1, B]
        result = torch.gather(samples, dim=0, index=indices)  # [1, B]
        return result.squeeze(0)  # [B, ]

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(mixture={self.mixture_distribution}, ' \
               f'component={self.component_distribution})'
