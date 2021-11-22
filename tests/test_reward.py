import numpy as np
from ase import Atoms

from molgym.rl.reward import InteractionReward


def test_calculation():
    reward_fn = InteractionReward()
    reward, _ = reward_fn.calculate(Atoms('H'))
    assert np.isclose(reward, 0.0)


def test_h2():
    atoms = Atoms('HH', positions=[(0, 0, 0), (1, 0, 0)])
    reward_fn = InteractionReward()
    reward, _ = reward_fn.calculate(atoms)
    assert np.isclose(reward, 0.1696435)
