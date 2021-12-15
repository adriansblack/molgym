import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional
from torch import nn, optim

from molgym.tools import CheckpointState, CheckpointHandler, masked_softmax, to_one_hot


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


def test_save_load():
    model = MyModel()
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag='test', keep=True)
        handler.save(state=CheckpointState(model, optimizer, lr_scheduler), counter=50)

        optimizer.step()
        lr_scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]['lr'], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, lr_scheduler))
        assert np.isclose(optimizer.param_groups[0]['lr'], initial_lr)


def test_one_hot():
    positions = np.array([[1], [3], [2]])
    indices = torch.from_numpy(positions)

    result = to_one_hot(indices=indices, num_classes=4).detach()
    expected = [
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    assert np.allclose(expected, result)


def test_one_hot_wrong_index():
    positions = np.array([
        [5],
    ])
    indices = torch.from_numpy(positions)

    with pytest.raises(RuntimeError):
        to_one_hot(indices=indices, num_classes=3).detach()


def test_softmax():
    logits = torch.from_numpy(np.array([
        [0.5, 0.5],
        [1.0, 0.5],
    ], dtype=float))

    mask_1 = torch.ones(size=logits.shape, dtype=torch.bool)

    y1 = masked_softmax(logits=logits, mask=mask_1)
    assert y1.shape, (2, 2)
    assert np.isclose(y1.sum().item(), 2.0)

    mask_2 = torch.from_numpy(np.array([[1, 0], [1, 0]], dtype=bool))
    y2 = masked_softmax(logits=logits, mask=mask_2)

    total = y2.sum(dim=0, keepdim=False)
    assert np.allclose(total, np.array([2, 0]))
