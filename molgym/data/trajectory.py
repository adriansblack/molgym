from dataclasses import dataclass
from typing import Iterable, Sequence, List, Optional, Dict, Union

import ase.data
import numpy as np

from .utils import Labels, Positions, SymbolTable

POSITIONS_KEY = 'positions'
ELEMENTS_KEY = 'elements'
BAG_KEY = 'bag'

FOCUS_KEY = 'focus'
ELEMENT_KEY = 'element'
DISTANCE_KEY = 'distance'
ORIENTATION_KEY = 'orientation'

Bag = np.ndarray

def bag_from_symbols(symbols: Iterable[str], s_table: SymbolTable) -> Bag:
    bag = [0] * len(s_table)
    for symbol in symbols:
        bag[s_table.symbol_to_element(symbol)] += 1

    return np.array(bag, dtype=int)

def bag_from_atom_costs(atom_costs: Dict[str,float], s_table: SymbolTable) -> Bag:
    bag = np.zeros((2,len(atom_costs)))
    for atom,cost in atom_costs.items():
        col = s_table.symbol_to_element(atom)
        if atom=='X': bag[0,col] = 0
        else: bag[0,col] = 1
        bag[1,col] = float(cost)

    return bag


def bag_from_symbol_count_dict(symbol_count_dict: Dict[str, int], z_table: SymbolTable) -> Bag:
    bag = [0] * len(z_table)
    for symbol, count in symbol_count_dict.items():
        bag[z_table.symbol_to_element(symbol)] = count

    return np.array(bag, dtype=int)


def remove_atom_from_bag(e: int, bag: Bag) -> Bag:
    if bag[e] < 1:
        raise ValueError(f"Cannot remove element (index) '{e}' from '{bag}'")

    copy = bag.copy()
    copy[e] -= 1
    return copy


def add_atom_to_bag(label: int, bag: Bag) -> Bag:
    copy = bag.copy()
    copy[label] += 1
    return copy


def bag_is_empty(bag: Bag) -> bool:
    return np.all(bag < 1)  # type: ignore


def no_real_atoms_in_bag(bag: Bag) -> bool:
    return np.all(bag[1:] < 1)  # type: ignore


@dataclass
class Action:
    focus: int
    element: int  # index, not Z
    distance: float
    orientation: np.ndarray


@dataclass
class State:
    elements: Labels  # labels (indices), not Zs
    positions: Positions
    bag: Bag


@dataclass
class SARS:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


Trajectory = Sequence[SARS]


def propagate(state: State, action: Action, infbag: bool = False) -> State:
    if infbag: 
        bag = state.bag
    else:
        bag = remove_atom_from_bag(action.element, state.bag)

        # If bag is empty, add sentinel element at position 0
        if bag_is_empty(bag):
            bag = add_atom_to_bag(label=0, bag=bag)
            
    if len(state.elements) == 1 and state.elements[0] == 0:
        return State(
            elements=np.array([action.element], dtype=int),
            positions=np.zeros((1, 3), dtype=float),
            bag=bag,
        )

    new_position = state.positions[action.focus] + action.distance * action.orientation
    return State(
        elements=np.concatenate([state.elements, np.array([action.element])]),
        positions=np.concatenate([state.positions, np.expand_dims(new_position, axis=0)], axis=0),
        bag=bag,
    )


def state_to_atoms(state: State, s_table: SymbolTable, info: Optional[Dict] = None, infbag=False) -> ase.Atoms:
    if infbag: 
        d = {BAG_KEY: {s_table.element_to_symbol(i): [int(v),cost] for i, (v,cost) in enumerate(state.bag.T)}}
        if s_table.element_to_symbol(state.elements[-1])=='Z': state.elements[-1]=s_table.symbol_to_element('X')
    else: d = {BAG_KEY: {s_table.element_to_symbol(i): int(v) for i, v in enumerate(state.bag)}}
    if info is not None:
        d.update(info)
    return ase.Atoms(
        symbols=[s_table.element_to_symbol(e) for e in state.elements],
        positions=state.positions,
        info=d,
    )


def state_from_atoms(atoms: ase.Atoms, s_table: SymbolTable) -> State:
    if len(atoms) == 0:
        elements = np.array([0], dtype=int)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    else:
        elements = np.array([s_table.symbol_to_element(s) for s in atoms.symbols])
        positions = atoms.positions

    if BAG_KEY in atoms.info:
        bag = bag_from_symbol_count_dict(atoms.info[BAG_KEY], z_table=s_table)
    else:
        bag = bag_from_symbols(symbols=['X'], s_table=s_table)

    return State(elements=elements, positions=positions, bag=bag)

def state_from_atom_costs(atoms: ase.Atoms, atom_costs: Dict[str,float], s_table: SymbolTable) -> State:
    if len(atoms) == 0:
        elements = np.array([0], dtype=int)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    else:
        elements = np.array([s_table.symbol_to_element(s) for s in atoms.symbols])
        positions = atoms.positions

    atom_costs['Z']=0.0
    bag = bag_from_atom_costs(atom_costs, s_table=s_table)

    return State(elements=elements, positions=positions, bag=bag)


def get_action(
    state: State,
    element: int,
    position: np.ndarray,
    focus: Optional[int],
) -> Action:
    assert len(state.elements) > 0

    # Replace dummy atom
    if len(state.elements) == 1 and state.elements[0] == 0:
        return Action(focus=0, element=element, distance=1.5, orientation=np.array([1.0, 0.0, 0.0]))

    if focus is None:
        distance = np.linalg.norm(state.positions - position, axis=-1, keepdims=False)  # [n_atoms, ]
        focus = np.argmin(distance).item()

    distance = np.linalg.norm(state.positions[focus] - position)
    orientation = -state.positions[focus] + position
    orientation /= np.linalg.norm(orientation)

    return Action(
        focus=focus,
        element=element,
        distance=distance,
        orientation=orientation,
    )


def rewind_state(s: State, index: int, infbag : bool = False) -> State:
    if index == 0:
        elements = np.array([0], dtype=int)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    else:
        elements = s.elements[:index]
        positions = s.positions[:index]

    # Place remaining atoms into bag
    bag = s.bag
    if not infbag:
        for e in s.elements[index:]:
            bag = add_atom_to_bag(e, bag)

        if not no_real_atoms_in_bag(bag) and bag[0] != 0:
            bag = remove_atom_from_bag(0, bag)

    return State(elements, positions, bag)


def generate_sparse_reward_trajectory(
    terminal_state: State,
    final_reward: float,
    start_index: int = 0,
    focuses: Optional[List[Optional[int]]] = None,
    infbag: bool = False
) -> Trajectory:
    assert 0 <= start_index <= len(terminal_state.elements)

    if focuses is None:
        focuses = [None] * (len(terminal_state.elements) - start_index)
    else:
        assert len(terminal_state.elements) - start_index == len(focuses)

    tau = []
    state = rewind_state(terminal_state, index=start_index, infbag=infbag)
    for i, focus in zip(range(start_index, len(terminal_state.elements)), focuses):
        action = get_action(
            state=state,
            focus=focus,
            element=terminal_state.elements[i],
            position=terminal_state.positions[i],
        )
        next_state = rewind_state(terminal_state, index=i + 1, infbag=infbag)
        tau.append(
            SARS(
                state=state,
                action=action,
                reward=final_reward if i == len(terminal_state.elements) - 1 else 0.0,
                next_state=next_state,
                done=i == len(terminal_state.elements) - 1,
            ))

        state = next_state

    return tau


def analyze_trajectory(tau: Trajectory) -> Dict[str, Union[int, float]]:
    return {
        'length': len(tau),
        'return': sum(sars.reward for sars in tau),
    }


def analyze_trajectories(taus: List[Trajectory]) -> Dict[str, float]:
    dicts = [analyze_trajectory(tau) for tau in taus]

    if len(dicts) == 0:
        return {}

    keys = dicts[0].keys()
    assert all(d.keys() == keys for d in dicts)
    return {key: np.mean([d[key] for d in dicts]) for key in keys}
