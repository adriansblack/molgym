import os

import ase.io
import pkg_resources
import torch_geometric
from e3nn import o3

from molgym import data, rl

resources_path = pkg_resources.resource_filename(__package__, 'resources')


def test_q():
    atoms = ase.io.read(filename=os.path.join(resources_path, 'h2o.xyz'), format='xyz', index=0)
    z_table = data.AtomicNumberTable([0, 1, 6, 8])
    tau = data.generate_sparse_reward_trajectory(atoms, z_table, final_reward=1.0)

    cutoff = 1.7
    q = rl.QFunction(
        r_max=cutoff,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps('3x0e + 3x1o + 3x2e + 3x3o'),
        network_width=32,
    )

    data_loader = torch_geometric.loader.DataLoader(
        dataset=[data.build_state_action_data(state=sars.state, cutoff=cutoff, action=None) for sars in tau],
        batch_size=5,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(data_loader))
    q_values = q(batch)
    assert q_values.shape == (len(tau), )
