import numpy as np
import pytest
from ase import Atoms

from molgym import data
from molgym.data import SymbolTable, Action
from molgym.rl import DiscreteMolecularEnvironment, SparseInteractionReward


def test_addition():
    reward_fn = SparseInteractionReward()
    s_table = SymbolTable('XHCO')
    terminal_state = data.state_from_atoms(atoms=Atoms('H2CO'), s_table=s_table)
    initial_state = data.rewind_state(terminal_state, index=0)
    env = DiscreteMolecularEnvironment(reward_fn=reward_fn, initial_state=initial_state, s_table=s_table)

    # Valid action
    next_state, reward, done, _ = env.step(
        Action(focus=0, element=1, distance=0.0, orientation=np.array([1.0, 0.0, 0.0])))

    assert next_state.elements[0] == 1
    assert np.isclose(reward, 0.0)
    assert not done

    # Invalid actions
    with pytest.raises(ValueError):
        env.step(Action(focus=0, element=0, distance=0.0, orientation=np.array([1.0, 0.0, 0.0])))

    with pytest.raises(ValueError):
        env.step(Action(focus=1, element=0, distance=0.0, orientation=np.array([1.0, 0.0, 0.0])))


def test_solo_distance():
    reward_fn = SparseInteractionReward()
    z_table = SymbolTable('XH')
    terminal_state = data.state_from_atoms(atoms=Atoms('H2'), s_table=z_table)
    initial_state = data.rewind_state(terminal_state, index=0)
    env = DiscreteMolecularEnvironment(reward_fn=reward_fn, initial_state=initial_state, s_table=z_table)

    # First H can be on its own
    _next_state, _reward, done, _info = env.step(
        Action(focus=0, element=1, distance=0.0, orientation=np.array([1.0, 0.0, 0.0])))
    assert not done

    # Second H cannot be alone
    _next_state, _reward, done, _info = env.step(
        Action(focus=0, element=1, distance=0.0, orientation=np.array([1.0, 0.0, 0.0])))
    assert done
