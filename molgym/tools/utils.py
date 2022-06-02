import json
import logging
import os
import sys
from typing import Iterable, Union, Optional, Dict, Any, Sequence, Tuple, List

import numpy as np
import torch

from .torch_tools import to_numpy


def get_tag(name: str, seed: int) -> str:
    return f'{name}_run-{seed}'


def setup_logger(level: Union[int, str] = logging.INFO, tag: Optional[str] = None, directory: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + '.log')
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


def get_optimizer(name: str, learning_rate: float, parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    if name == 'adam':
        amsgrad = False
    elif name == 'amsgrad':
        amsgrad = True
    else:
        raise RuntimeError(f"Unknown optimizer '{name}'")

    return torch.optim.Adam(parameters, lr=learning_rate, amsgrad=amsgrad)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + '.txt'
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        logging.debug(f'Saving info: {self.path}')
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode='a', encoding='ascii') as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write('\n')


def random_train_valid_split(items: Sequence, valid_fraction: float, seed: int) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return [items[i] for i in indices[:train_size]], [items[i] for i in indices[train_size:]]

def process_symbol_costs_str(symbol_costs):
    def cost_se(cost_se_str):
        cost,se = cost_se_str.split('@')
        cost = float(cost)
        if ':' in se:
            s,e = se.split(':')
            s,e = int(s), int(e)
        else: 
            s = int(se)
            e = s
        return (cost,s,e)
    def split_counts(atom_str):
        atom,cost_se_str = atom_str.split('=')
        costs = list(map(cost_se,cost_se_str.split(','))) 
        return [atom,costs]
    return dict(map(split_counts, symbol_costs.split(' ')))
