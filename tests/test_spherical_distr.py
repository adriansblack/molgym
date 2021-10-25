import numpy as np
import torch

from molgym.spherical_distrs import SphericalUniform, SO3Distribution, generate_fibonacci_grid
from molgym.tools import to_numpy


def test_uniform_shape():
    num_samples = 1000
    distr = SphericalUniform()
    samples = distr.sample(torch.Size((num_samples, )))
    assert samples.shape == (num_samples, 3)


def test_uniform_min_max():
    distr = SphericalUniform(batch_shape=(3, ))
    assert distr.get_max_prob().shape == (3, )


def test_uniform_distance():
    samples = SphericalUniform().sample(torch.Size((1_000, )))
    assert np.allclose(samples.norm(dim=-1), 1.)


def test_uniform_mean():
    torch.manual_seed(1)
    distr = SphericalUniform()
    samples = distr.sample(torch.Size((200_000, )))
    assert np.isclose(samples.mean(dim=0).norm().item(), 0, atol=1E-2)


def test_uniform_argmax():
    distr = SphericalUniform(batch_shape=torch.Size((3, )))
    arg_maxes = distr.argmax()
    assert arg_maxes.shape == distr.batch_shape + distr.event_shape


def test_so3distr_max():
    lmax = 3
    a_lms = torch.randn(size=torch.Size((5, (lmax + 1)**2)))
    distr = SO3Distribution(a_lms=a_lms, lmax=lmax, gamma=100)
    assert distr.get_max_log_prob().shape == (5, )


def cartesian_to_spherical(pos: np.ndarray) -> np.ndarray:
    theta_phi = np.empty(shape=pos.shape[:-1] + (2, ))

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)
    theta_phi[..., 0] = np.arccos(z / r)  # theta
    theta_phi[..., 1] = np.arctan2(y, x)  # phi

    return theta_phi


def test_so3distr_sample():
    torch.manual_seed(1)

    samples_shape = (2048, )
    lmax = 3
    gamma = 25
    a_lms = torch.randn(size=torch.Size((2, (lmax + 1)**2)))

    distr = SO3Distribution(a_lms=a_lms, lmax=lmax, gamma=gamma)

    samples = distr.sample(samples_shape)
    assert samples.shape == samples_shape + distr.batch_shape + distr.event_shape

    angles = cartesian_to_spherical(to_numpy(samples))  # [S, B, 2]
    mean_angles = np.mean(angles, axis=0)  # [B, 2]
    assert mean_angles.shape == (2, 2)

    distr_1 = SO3Distribution(a_lms=a_lms[[0]], lmax=lmax, gamma=gamma)
    samples_1 = distr_1.sample(samples_shape)
    angles_1 = cartesian_to_spherical(to_numpy(samples_1))  # [S, 1, 2]
    mean_angles_1 = np.mean(angles_1, axis=0)  # [1, 2]

    distr_2 = SO3Distribution(a_lms=a_lms[[1]], lmax=lmax, gamma=gamma)
    samples_2 = distr_2.sample(samples_shape)
    angles_2 = cartesian_to_spherical(to_numpy(samples_2))  # [S, 1, 2]
    mean_angles_2 = np.mean(angles_2, axis=0)  # [1, 2]

    # Assert that batching does not affect the result
    assert np.allclose(mean_angles[0], mean_angles_1, atol=0.1)
    assert np.allclose(mean_angles[1], mean_angles_2, atol=0.1)


def test_so3distr_prob():
    lmax = 3
    gamma = 25
    a_lms = torch.randn(size=torch.Size((5, (lmax + 1)**2)))
    distr = SO3Distribution(a_lms=a_lms, lmax=lmax, gamma=gamma)
    samples = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    assert distr.log_prob(samples.unsqueeze(1)).shape == (3, 5)


def test_so3distr_max_sample():
    lmax = 3
    gamma = 25
    a_lms = torch.randn(size=torch.Size((5, (lmax + 1)**2)))
    distr = SO3Distribution(a_lms=a_lms, lmax=lmax, gamma=gamma)
    samples = distr.argmax(count=17)
    assert samples.shape == (5, 3)


def test_so3distr_normalization():
    lmax = 3
    gamma = 25
    a_lms = torch.randn(size=torch.Size((5, (lmax + 1)**2)))
    distr = SO3Distribution(a_lms=a_lms, lmax=lmax, gamma=gamma)
    grid = generate_fibonacci_grid(n=1024)
    grid_t = torch.tensor(grid, dtype=torch.float).unsqueeze(1)
    probs = torch.exp(distr.log_prob(grid_t))
    integral = 4 * np.pi * torch.mean(probs, dim=0)

    assert np.allclose(to_numpy(integral), 1.0, atol=5e-3)
