import abc
import logging
from typing import Tuple, List

import ase.data
import numpy as np

from molgym import data
from molgym.data import Action, State, AtomicNumberTable, DiscreteBagState
from .reward import SparseInteractionReward


def is_terminal(state: State) -> bool:
    if not isinstance(state, DiscreteBagState):
        return False

    return data.bag_is_empty(state.bag)


def any_too_close(
    positions: np.ndarray,  # [n1, 3]
    other_positions: np.ndarray,  # [n2, 3]
    threshold: float,
) -> bool:
    positions = np.expand_dims(positions, axis=1)
    other_positions = np.expand_dims(other_positions, axis=0)
    return bool(np.any(np.linalg.norm(positions - other_positions) < threshold))


class MolecularEnvironment(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        raise NotImplementedError


class DiscreteMolecularEnvironment(MolecularEnvironment):
    def __init__(
            self,
            reward_fn: SparseInteractionReward,
            initial_state: DiscreteBagState,
            z_table: AtomicNumberTable,
            min_atomic_distance=0.6,  # Angstrom
            max_solo_distance=2.0,  # Angstrom
            min_reward=-0.6,  # Hartree
    ):
        self.reward_fn = reward_fn
        self.initial_state = initial_state
        self.z_table = z_table

        self.min_atomic_distance = min_atomic_distance
        self.max_solo_distance = max_solo_distance
        self.min_reward = min_reward

        self.candidate_elements = [
            self.z_table.z_to_index(ase.data.atomic_numbers[s]) for s in ['H', 'F', 'Cl', 'Br']
            if ase.data.atomic_numbers[s] in self.z_table.zs
        ]

        self.current_state = self.initial_state
        self.terminal = False

    def reset(self) -> State:
        self.current_state = self.initial_state
        self.terminal = False
        return self.current_state

    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        if self.terminal:
            raise RuntimeError('Stepping with terminal state')

        self.current_state = data.propagate_discrete_bag_state(self.current_state, action)

        # Is state valid?
        if not self._is_valid_state(self.current_state):
            self.terminal = True
            return self.current_state, self.min_reward, True, {}

        # Is state terminal?
        if is_terminal(self.current_state):
            done = True
            self.terminal = True
            reward, info = self._calculate_reward(self.current_state)
        else:
            done = False
            reward, info = 0.0, {}

        # Is reward too low?
        if reward < self.min_reward:
            done = True
            self.terminal = True
            reward = self.min_reward

        return self.current_state, reward, done, info

    def _is_valid_state(self, state: State) -> bool:
        if len(state.elements) <= 1:
            return True

        if any_too_close(state.positions[:-1], state.positions[-1:], threshold=self.min_atomic_distance):
            logging.debug('Two atoms are too close')
            return False

        if not self._last_covered(state):
            logging.debug('There is a single atom floating around')
            return False

        return True

    def _calculate_reward(self, state: State) -> Tuple[float, dict]:
        atoms = data.state_to_atoms(state, self.z_table)
        return self.reward_fn.calculate(atoms)

    def _last_covered(self, state: State) -> bool:
        # Ensure that certain atoms are not too far away from the nearest heavy atom to avoid H2, F2,... formation

        if len(state.elements) <= 1:
            return True

        last_element, last_position = state.elements[-1], state.positions[-1]
        if last_element not in self.candidate_elements:
            return True

        for other_element, other_position in zip(state.elements[:-1], state.positions[:-1]):
            if other_element in self.candidate_elements:
                continue

            distance = np.linalg.norm(other_position - last_position)
            if distance < self.max_solo_distance:
                return True

        return False


class EnvironmentCollection:
    def __init__(self, envs: List[MolecularEnvironment]) -> None:
        self.envs = envs

    def __len__(self):
        return len(self.envs)

    def step(self, actions: List[Action]) -> List[Tuple[State, float, bool, dict]]:
        assert len(self.envs) == len(actions)
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def reset_all(self) -> List[State]:
        return [env.reset() for env in self.envs]

    def reset_env(self, index: int) -> State:
        return self.envs[index].reset()
