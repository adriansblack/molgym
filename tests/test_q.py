import os

import ase.io
import pkg_resources
from e3nn import o3

from molgym import data, rl

resources_path = pkg_resources.resource_filename(__package__, 'resources')


def test_q():
    atoms = ase.io.read(filename=os.path.join(resources_path, 'h2o.xyz'), format='xyz', index=0)
    s_table = data.SymbolTable('XHCO')
    terminal_state = data.state_from_atoms(atoms=atoms, s_table=s_table)
    tau = data.generate_sparse_reward_trajectory(terminal_state, final_reward=1.0)

    cutoff = 1.7
    q = rl.QFunction(
        r_max=cutoff,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        num_interactions=2,
        num_elements=len(s_table),
        hidden_irreps=o3.Irreps('3x0e + 3x1o + 3x2e + 3x3o'),
        network_width=32,
    )

    data_loader = data.DataLoader(
        dataset=[data.process_sa(state=sars.state, cutoff=cutoff, action=None) for sars in tau],
        batch_size=5,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(data_loader))
    q_values = q(batch['state'])
    assert q_values.shape == (len(tau), )
