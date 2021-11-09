from abc import ABC
from typing import Dict, Any

import numpy as np
import quadpy
import torch
from e3nn import o3


def generate_fibonacci_grid(n: int) -> np.ndarray:
    # Based on: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    golden_ratio = (1 + 5**0.5) / 2
    offset = 0.5

    index = np.arange(0, n)
    theta = np.arccos(1 - 2 * (index + offset) / n)
    phi = 2 * np.pi * index / golden_ratio

    theta_phi = np.stack([theta, phi], axis=-1)

    return spherical_to_cartesian(theta_phi)


def spherical_to_cartesian(theta_phi: np.ndarray) -> np.ndarray:
    theta, phi = theta_phi[..., 0], theta_phi[..., 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


class SphericalDistribution(torch.distributions.Distribution, ABC):
    arg_constraints: Dict[str, Any] = {}
    has_rsample = False

    def __init__(self, batch_shape=torch.Size(), validate_args=None) -> None:
        super().__init__(batch_shape, event_shape=torch.Size((3, )), validate_args=validate_args)

    @staticmethod
    def _spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)


class SphericalUniform(SphericalDistribution):
    def __init__(self, batch_shape=torch.Size()) -> None:
        super().__init__(batch_shape, validate_args=False)
        self.uniform_dist = torch.distributions.Uniform(0.0, 1.0)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        # Based on: http://corysimon.github.io/articles/uniformdistn-on-sphere/
        # Get shape
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self._batch_shape

        # Sample from transformed uniform
        theta = torch.acos(1 - 2 * self.uniform_dist.sample(shape))
        phi = 2 * np.pi * self.uniform_dist.sample(shape)

        # Convert to Cartesian coordinates
        return self._spherical_to_cartesian(theta=theta, phi=phi)

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        return torch.ones(size=value.shape[:-1], device=value.device) / (4 * np.pi)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob(value).clamp(min=1e-10))

    def get_max_prob(self) -> torch.Tensor:
        return torch.ones(size=self.batch_shape) / (4 * np.pi)

    def argmax(self) -> torch.Tensor:
        return self.sample()


class SO3Distribution(SphericalDistribution):
    def __init__(self, a_lms: torch.Tensor, lmax: int, gamma: float) -> None:
        assert len(a_lms.shape) == 2
        assert a_lms.shape[-1] == (lmax + 1)**2

        super().__init__(batch_shape=torch.Size(a_lms.shape[:-1]), validate_args=False)

        self.b_lms = self.normalize(a_lms)
        self.lmax = lmax
        self.gamma = gamma
        self.device = a_lms.device

        self.spherical_uniform = SphericalUniform(batch_shape=self.batch_shape)
        self.uniform_dist = torch.distributions.Uniform(low=0.0, high=1.0, validate_args=False)
        self.log_z = self.compute_log_z()

    def __str__(self) -> str:
        return f'SO3Distribution(l_max={self.lmax}, gamma={self.gamma}, batch_shape={tuple(self.batch_shape)})'

    @staticmethod
    def normalize(a_lms) -> torch.Tensor:
        k = torch.sum(torch.square(a_lms), dim=-1, keepdim=True)  # [B, 1]
        return a_lms / torch.sqrt(k.clamp(min=1e-10))

    def get_max_log_prob(self) -> torch.Tensor:
        # grid_points: [S, B=1, 3]
        grid_points = torch.tensor(generate_fibonacci_grid(n=4096), device=self.device).unsqueeze(1)
        log_probs = self.log_prob(grid_points)  # [S, B]

        # Maximum over grid points
        maximum, _ = torch.max(log_probs, dim=0)

        return maximum  # [B, ]

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        # Note: [S, B, E]
        assert len(self.batch_shape) == 1
        num_batches = self.batch_shape[0]

        max_log_prob = self.get_max_log_prob()
        max_log_prob_proposal = torch.log(self.spherical_uniform.get_max_prob()).to(self.device)

        log_m_value = (max_log_prob - max_log_prob_proposal).unsqueeze(0)  # [S=1, B]

        m_value = torch.exp(log_m_value.clamp(-8, 8))
        count = min(max(1, int(2 * torch.max(m_value).item())), 1024)

        # number of samples per batch item
        num_samples = int(np.product(sample_shape))

        accepted_t = torch.empty(size=(0, num_batches), dtype=torch.bool, device=self.device)
        candidates_t = torch.empty(size=(0, num_batches) + self.event_shape, device=self.device)

        while torch.any(accepted_t.sum(dim=0) < num_samples):
            candidates = self.spherical_uniform.sample(torch.Size((count, ))).to(self.device)  # [count, E]
            log_threshold = self.log_prob(candidates) - log_m_value - self.spherical_uniform.log_prob(candidates)
            u = self.uniform_dist.sample(torch.Size((count, ))).unsqueeze(1).to(self.device)  # [count, 1]
            accepted = u < torch.exp(log_threshold)  # [count, B]

            accepted_t = torch.cat([accepted_t, accepted], dim=0)
            candidates_t = torch.cat([candidates_t, candidates], dim=0)

        # Collect accepted samples
        samples = []
        for i in range(num_batches):
            cs = candidates_t[:, i]  # [count, E]
            acs = accepted_t[:, i]  # [count, ]
            samples.append(cs[acs][:num_samples])

        samples_t = torch.stack(samples, dim=0)  # [B, S, E]
        return samples_t.transpose(0, 1).reshape(sample_shape + self.batch_shape + self.event_shape).contiguous()

    def argmax(self, count=128) -> torch.Tensor:
        samples = self.sample(sample_shape=torch.Size((count, )))  # [S, B, 3]
        log_probs_unnormalized = self.log_prob_unnormalized(samples)  # [S, B]
        indices = torch.argmax(log_probs_unnormalized, dim=0)  # [B, ]
        gather_indices = indices.unsqueeze(0).unsqueeze(-1).expand((-1, -1) + self.event_shape)  # [1, B, 3]
        result = torch.gather(samples, dim=0, index=gather_indices)  # [1, B, 3]
        return result.squeeze(0)  # squeeze out samples dimension

    def log_prob_unnormalized(
            self,
            pos: torch.Tensor,  # [S, B, 3]
    ) -> torch.Tensor:  # [S, B]
        y_lms = o3.spherical_harmonics(
            l=list(range(0, self.lmax + 1)),
            x=pos,
            normalize=True,  # normalize position vectors
            normalization='integral',  # normalization of spherical harmonics
        )  # [S, B, (lmax + 1)^2]

        # b_lms: [B, (lmax + 1)^2]
        # y_lms: [S, B, (lmax + 1)^2]
        return self.gamma * torch.sum(self.b_lms * y_lms, dim=-1)

    def compute_log_z(self) -> torch.Tensor:
        grid = quadpy.u3._lebedev.lebedev_071()  # pylint: disable=protected-access
        grid_points = torch.tensor(grid.points.transpose(), device=self.device).unsqueeze(1)  # grid_points: [S, 1, 3]
        log_probs_unnormalized = self.log_prob_unnormalized(grid_points)  # [S, B]
        weights = torch.tensor(grid.weights, device=self.device).unsqueeze(-1)  # [S, 1]
        return np.log(4 * np.pi) + torch.logsumexp(log_probs_unnormalized + torch.log(weights), dim=0)

    def log_prob(self, value: torch.Tensor):
        return self.log_prob_unnormalized(value) - self.log_z
