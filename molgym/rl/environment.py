import abc
import logging
from typing import Tuple, List
from collections import Counter
import numpy as np

from molgym import data
from molgym.data import Action, State, SymbolTable
from .reward import SparseInteractionReward


def is_terminal(state: State, infbag : bool = False, stop_idx: int = None) -> bool:
    if infbag: 
        if len(state.elements)>=10: return True
        return state.elements[-1]==stop_idx #Matches 'Z'
    else: return data.no_real_atoms_in_bag(state.bag)


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

class atom_price_schedule:
    def __init__(self, atom: str):
        self.schedule = []
        self.prices = []
        self.element = atom
    def add(self, count: int, cost: float):
        self.prices.append(cost)
        self.schedule.append(count)
    

class DiscreteMolecularEnvironment(MolecularEnvironment):
    def __init__(
            self,
            reward_fn: SparseInteractionReward,
            initial_state: State,
            s_table: SymbolTable,
            min_atomic_distance=0.6,  # Angstrom
            max_solo_distance=2.0,  # Angstrom
            min_reward=-0.6,  # Hartree
            infbag: bool = False,
            stop_idx: int = None,
            costs_sched: dict = None,
            seed: int = 0,
            byiter: bool = True,
            maskZ: bool = False,
    ):
        self.reward_fn = reward_fn
        self.initial_state = initial_state
        self.s_table = s_table

        self.min_atomic_distance = min_atomic_distance
        self.max_solo_distance = max_solo_distance
        self.min_reward = min_reward

        self.candidate_elements = [
            self.s_table.symbol_to_element(s) for s in ['H', 'F', 'Cl', 'Br'] if s in self.s_table.symbols
        ]

        self.current_state = self.initial_state
        self.terminal = False

        self.infbag = infbag
        self.stop_idx = stop_idx
        self.cost_sched = costs_sched
        self.byiter = byiter
        if self.infbag: 
            max_incr_iter = 0
            for cost_tups in costs_sched.values():
                max_incr_iter = max(max_incr_iter,np.sum(np.array(cost_tups,dtype=int)[:,2]))
            self.bag_schedule = np.zeros((max_incr_iter+1,len(costs_sched)))
        self.maskZ = maskZ

        self.rng = np.random.default_rng(seed)

    def reset(self) -> State:
        self.current_state = self.initial_state
        self.terminal = False
        if self.infbag: self.reset_bag_schedule()
        return self.current_state

    def reset_bag_schedule(self):
        for atom,cost_tups in self.cost_sched.items():
            incr_idxs = []
            costs = []
            for (c,s,e) in cost_tups:
                costs.append(c)
                if s==e: r = s
                else: r = self.rng.integers(s,e,endpoint=True)
                incr_idxs.append(r)
            incr_idxs.append(1)
            incr_idxs = incr_idxs[1:]
            y = np.repeat(costs,incr_idxs)
            y = np.pad(y,[0,len(self.bag_schedule)-len(y)],mode='edge')
            self.bag_schedule[:,self.s_table.symbol_to_element(atom)]=y

    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        if self.terminal:
            raise RuntimeError('Stepping with terminal state')

        if self.infbag: new_bag = data.next_bag(self.byiter, self.current_state.bag,self.current_state.elements, self.bag_schedule,action.element, self.stop_idx, self.maskZ)
        self.current_state = data.propagate(self.current_state, action, self.infbag, new_bag)

        # Is state valid?
        if not self._is_valid_state(self.current_state):
            self.terminal = True
            return self.current_state, self.min_reward-1, True, {}

        # Is state terminal?
        if is_terminal(self.current_state, self.infbag, self.stop_idx):
            done = True
            self.terminal = True

            if self.infbag: rew_state = data.rewind_state(self.current_state, -1, self.infbag, self.byiter, self.bag_schedule)
            else: rew_state = self.current_state
            
            if self.infbag and len(rew_state.elements)==0: 
                return self.current_state, self.min_reward-1, True, {}
            else: reward, info = self._calculate_reward(rew_state)

            if self.infbag: 
                if self.byiter: 
                    idx = [np.arange(len(self.current_state.elements)),self.current_state.elements]
                    if len(self.current_state.elements)>len(self.bag_schedule):
                        extra = len(self.current_state.elements)-len(self.bag_schedule)
                        bag_schedule_temp = np.vstack((self.bag_schedule,np.tile(self.bag_schedule[-1],(extra,1))))
                    else: bag_schedule_temp = self.bag_schedule
                    reward +=np.sum(bag_schedule_temp[idx])
                else:
                    for elem,count in Counter(self.current_state.elements).items():
                        reward += np.sum(self.bag_schedule[:count,elem])
                        extra = count-len(self.bag_schedule)
                        if extra>0: reward+= self.bag_schedule[-1,elem]*extra
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
        # Check bag
        if not self.infbag and any(item < 0 for item in state.bag):
            return False

        # Check canvas
        if len(state.elements) <= 1:
            return True

        if any_too_close(state.positions[:-1], state.positions[-1:], threshold=self.min_atomic_distance):
            logging.debug('Two atoms are too close')
            return False

        # if not self._last_covered(state):
        #     logging.debug('There is a single atom floating around')
        #     return False

        return True

    def _calculate_reward(self, state: State) -> Tuple[float, dict]:
        return self.reward_fn.calculate(symbols=[self.s_table.element_to_symbol(e) for e in state.elements],
                                        positions=state.positions,
                                        gradients=False)

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
