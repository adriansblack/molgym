import numpy as np
import torch_geometric
from e3nn import o3

from molgym import data, modules, tools

config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array([
        [1.1, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [-1.0, 0.5, 0.0],
    ]),
    forces=np.array([
        [0.0, -1.3, 0.0],
        [1.0, 0.2, 0.0],
        [0.0, 1.1, 0.3],
    ]),
    energy=-1.5,
)

table = data.AtomicNumberTable([1, 8])


def test_bo_model():
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    model = modules.EnergyModel(
        r_max=2.0,
        num_bessel=7,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_elements=len(table),
        num_interactions=2,
        atomic_energies=atomic_energies,
        hidden_irreps=o3.Irreps('10x0e + 10x0o + 8x1e + 8x1o + 4x2e + 4x2o'),
    )

    assert tools.count_parameters(model) == 3164

    atomic_data = data.build_energy_forces_data(config, z_table=table, cutoff=3.0)
    data_loader = torch_geometric.loader.DataLoader([atomic_data, atomic_data], batch_size=2)
    batch = next(iter(data_loader))

    output = model(batch)
    assert output['energy'].shape == (2, )
    assert output['forces'].shape == (6, 3)
