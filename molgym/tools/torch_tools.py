import collections
import logging
from typing import Dict, Sequence, Union

import numpy as np
import torch
import torch_geometric
import torch_scatter

TensorDict = Dict[str, torch.Tensor]


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>

    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes, )
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch_scatter.composite.scatter_softmax(src=logits, index=mask.to(torch.long), dim=-1) * mask


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def detach_tensor_dict(td: TensorDict) -> TensorDict:
    return {k: v.detach() for k, v in td.items()}


def dict_to_device(o: collections.Mapping, device: torch.device) -> Dict:
    d = {}
    for k, v in o.items():
        if isinstance(v, (torch.Tensor, torch_geometric.data.Data)):
            d[k] = v.to(device)
        elif isinstance(v, collections.Mapping):
            d[k] = dict_to_device(v, device)
        else:
            raise NotImplementedError
    return d


def concat_tensor_dicts(tds: Sequence[TensorDict]) -> TensorDict:
    if len(tds) == 0:
        return {}

    return {k: torch.cat([td[k] for td in tds], dim=0) for k in tds[0].keys()}


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def init_device(device_str: str) -> torch.device:
    if device_str == 'cuda':
        assert (torch.cuda.is_available()), 'No CUDA device available!'
        logging.info(f'CUDA Device: {torch.cuda.current_device()}')
        torch.cuda.init()
        return torch.device('cuda')

    logging.info('Using CPU')
    return torch.device('cpu')


dtype_dict = {'float32': torch.float32, 'float64': torch.float64}


def set_default_dtype(dtype: str) -> None:
    torch.set_default_dtype(dtype_dict[dtype])
