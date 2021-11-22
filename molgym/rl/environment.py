import logging
from typing import Tuple

import ase.data
import numpy as np
from ase import Atoms, Atom

from molgym.data import Action, State, AtomicNumberTable, propagate_discrete_bag_state, DiscreteBagState, bag_is_empty, \
    state_to_atoms
from .reward import InteractionReward


def is_terminal(state: State) -> bool:
    if not isinstance(state, DiscreteBagState):
        return False

    return bag_is_empty(state.bag)


def any_too_close(
    positions: np.ndarray,  # [n1, 3]
    other_positions: np.ndarray,  # [n2, 3]
    threshold: float,
) -> bool:
    positions = np.expand_dims(positions, axis=1)
    other_positions = np.expand_dims(other_positions, axis=0)
    return any(np.linalg.norm(positions - other_positions) < threshold)


class DiscreteMolecularEnvironment:
    def __init__(
        self,
        reward: InteractionReward,
        initial_state: DiscreteBagState,
        z_table: AtomicNumberTable,
        min_atomic_distance=0.6,  # Angstrom
        max_solo_distance=2.0,  # Angstrom
        min_reward=-0.6,  # Hartree
        seed=0,
    ):
        self.reward = reward
        self.initial_state = initial_state
        self.z_table = z_table

        self.min_atomic_distance = min_atomic_distance
        self.max_solo_distance = max_solo_distance
        self.min_reward = min_reward

        self.candidate_indices = [z_table.z_to_index(ase.data.atomic_numbers[s]) for s in ['H', 'F', 'Cl', 'Br']]

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.current_state = self.initial_state
        self.terminal = False

    def seed(self, seed=None) -> int:
        self.seed = seed or np.random.randint(int(1e5))
        self.random_state = np.random.RandomState(self.seed)
        return self.seed

    def reset(self) -> None:
        self.random_state = np.random.RandomState(self.seed)
        self.current_state = self.initial_state
        self.terminal = False

    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        if self.terminal:
            raise RuntimeError('Stepping with terminal state')

        self.current_state = propagate_discrete_bag_state(self.current_state, action)

        if not self._is_valid_state(self.current_state):
            self.terminal = True
            return self.current_state, self.min_reward, True, {}

        # Check if state is terminal
        if is_terminal(self.current_state):
            done = True
            self.terminal = True
            reward, info = self._calculate_reward(self.current_state)

        else:
            done = False
            reward, info = 0.0, {}

        if reward < self.min_reward:
            done = True
            self.terminal = True
            reward = self.min_reward

        return self.current_state, reward, done, info

    def _is_valid_state(self, state: State) -> bool:
        if any_too_close(state.positions[:-1], state.positions[-1:], threshold=self.min_atomic_distance):
            return False

        return self._last_covered(state)

    def _calculate_reward(self, state: State) -> Tuple[float, dict]:
        atoms = state_to_atoms(state, self.z_table)
        return self.reward.calculate(atoms)

    def _last_covered(self, state: State) -> bool:
        # Ensure that certain atoms are not too far away from the nearest heavy atom to avoid H2, F2,... formation
        if len(existing_atoms) == 0 or new_atom.symbol not in candidates:
            return True

        for existing_atom in existing_atoms:
            if existing_atom.symbol in candidates:
                continue

            distance = np.linalg.norm(existing_atom.position - new_atom.position)
            if distance < self.max_solo_distance:
                return True

        logging.debug('There is a single atom floating around')
        return False