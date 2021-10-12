import numpy as np
import torch

from molgym.gmm import GaussianMixtureModel
from molgym.tools import to_numpy


def test_gmm():
    torch.manual_seed(1)

    log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.5, 0.5]]))
    means = torch.tensor([[-0.5, 0.3], [0.0, 0.2]])
    log_stds = torch.log(torch.tensor([[0.2, 0.5], [0.3, 0.2]]))
    gmm = GaussianMixtureModel(log_probs=log_probs, means=means, stds=torch.exp(log_stds))

    samples = gmm.sample(torch.Size((3, )))
    assert samples.shape == (3, 2)

    argmax = gmm.argmax(count=128)
    assert argmax.shape == (2, )
    assert np.allclose(to_numpy(argmax), np.array([-0.495, 0.156]), atol=1e-2)
