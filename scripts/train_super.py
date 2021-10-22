import argparse
import logging
from typing import Dict

import torch_geometric
from e3nn import o3

from molgym import tools, data
from molgym.data import graph_tools
from molgym.modules import Policy


def add_supervised(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--xyz', help='path to XYZ file', type=str, required=True)
    parser.add_argument('--z_energies', help='atomic energy of elements (e.g.,: z1:e1, z2:e2)', type=str, required=True)
    return parser


def parse_z_energies(z_energies: str) -> Dict[int, float]:
    d = {0: 0.0}
    for item in z_energies.split(','):
        z_str, e_str = item.split(':')
        d[int(z_str)] = float(e_str)

    return d


def compute_interaction_energy(config: data.Configuration, z_energies: Dict[int, float]) -> float:
    return config.energy - sum(z_energies[z] for z in config.atomic_numbers)


def main() -> None:
    parser = tools.build_default_arg_parser()
    parser = add_supervised(parser)
    args = parser.parse_args()

    # Setup
    tag = tools.get_tag(name=args.name, seed=args.seed)
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    logging.info(f'Configuration: {args}')
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    # Create energies and Z table
    z_energies = parse_z_energies(args.z_energies)
    logging.info('Atomic energies: {' + ', '.join(f'{k}: {v:.4f}' for k, v in z_energies.items()) + '}')
    z_table = data.AtomicNumberTable(sorted(z_energies.keys()))
    logging.info(z_table)

    # Load atoms list
    atoms_list = data.load_xyz(args.xyz)

    # Generate SARS list
    sars_list = []
    for atoms in atoms_list:
        e_inter = compute_interaction_energy(config=data.config_from_atoms(atoms), z_energies=z_energies)
        graph = graph_tools.generate_topology(atoms, cutoff_distance=args.d_max)

        num_paths = max(int(len(atoms) * args.num_paths_per_atom), 1)
        for seed in range(num_paths):
            sequence = graph_tools.breadth_first_rollout(graph, seed=seed)
            sars_list += data.generate_sparse_reward_trajectory(
                atoms=graph_tools.select_atoms(atoms, sequence),
                final_reward=e_inter,
                z_table=z_table,
            )

    geometric_data = [
        data.build_state_action_data(state=item.state,
                                     action=item.action,
                                     z_table=z_table,
                                     cutoff=args.d_max) for item in sars_list
    ]
    data_loader = torch_geometric.data.DataLoader(
        dataset=geometric_data,
        batch_size=17,
        shuffle=False,
        drop_last=False,
    )

    policy = Policy(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        network_width=32,
    )

    logging.info(policy)
    logging.info(f'Number of parameters: {tools.count_parameters(policy)}')

    for batch in data_loader:
        print(batch.num_nodes)
        output = policy(batch)
        for o in output:
            print(o.shape)
        break


if __name__ == '__main__':
    main()
