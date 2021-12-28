import argparse

import ase.data
import ase.io
import numpy as np
import torch
from matplotlib import pyplot as plt

from molgym import tools, rl, data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Agent Inspector')

    parser.add_argument('--model', help='path to model', type=str, required=True)
    parser.add_argument('--zs', help='atomic numbers in table', type=str, required=True)

    parser.add_argument('--d_min', help='minimum distance (in Ang)', type=float, default=0.9)
    parser.add_argument('--d_max', help='maximum distance (in Ang)', type=float, default=1.8)

    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--default_dtype',
                        help='set default dtype',
                        type=str,
                        choices=['float32', 'float64'],
                        default='float64')

    return parser.parse_args()


colors = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]


def main():
    args = parse_args()
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    # Load model and data
    model: rl.SACAgent = torch.load(f=args.model, map_location=device)

    # Parse Z table
    z_table = data.AtomicNumberTable([int(z) for z in args.zs.split(',')])

    def generate_config(d: float) -> ase.Atoms:
        positions = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        return ase.Atoms(symbols='OO', positions=positions)

    ds = np.linspace(args.d_min, args.d_max, num=50)
    atoms_traj = [generate_config(d) for d in ds]
    reward_fn = rl.SparseInteractionReward()
    rewards = [reward_fn.calculate(atoms)[0] for atoms in atoms_traj]

    # Single state
    geometric_data = [
        data.process_sa(
            state=data.get_state_from_atoms(atoms=atoms_traj[0], index=1, z_table=z_table),
            cutoff=args.d_max + 0.05,
            action=None,
        )
    ]
    data_loader = data.DataLoader(
        dataset=geometric_data,
        batch_size=len(geometric_data),
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(data_loader))
    q1_single = tools.to_numpy(model.q1(batch['state']))
    q2_single = tools.to_numpy(model.q2(batch['state']))
    print(q1_single, q2_single)

    # Multiple states
    states = [data.get_state_from_atoms(atoms=atoms, index=len(atoms), z_table=z_table) for atoms in atoms_traj]
    geometric_data = [data.process_sa(
        state=state,
        cutoff=args.d_max + 0.01,
        action=None,
    ) for state in states]

    data_loader = data.DataLoader(
        dataset=geometric_data,
        batch_size=len(geometric_data),
        shuffle=False,
        drop_last=False,
    )

    q1_list = []
    q2_list = []
    for batch in data_loader:
        q1_list.append(model.q1(batch['state']))
        q2_list.append(model.q2(batch['state']))

    q1s = tools.to_numpy(torch.cat(q1_list, dim=0))
    q2s = tools.to_numpy(torch.cat(q2_list, dim=0))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), constrained_layout=True)
    ax.plot(ds, rewards, color='black', label='Reward')
    ax.plot(ds, q1s, label='Q1')
    ax.plot(ds, q2s, label='Q2')
    ax.set_xlabel(r'Distance $d$ [Ang]')
    fig.legend()
    fig.show()


if __name__ == '__main__':
    main()
