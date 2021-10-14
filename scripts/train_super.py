import argparse
import logging
from typing import Dict

from molgym import tools, data


def add_supervised(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--xyz', help='path to XYZ file', type=str, required=True)
    parser.add_argument('--z_energies',
                        help='atomic energy of elements (e.g.,: z1:e1, z2:e2, )',
                        type=str,
                        required=True)
    return parser


def parse_z_energies(z_energies: str) -> Dict[int, float]:
    d = {}
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
        for seed in range(len(atoms) * args.num_paths_per_atom):
            sars_list += data.generate_sparse_reward_trajectory(
                atoms=data.reorder_breadth_first(atoms, cutoff_distance=args.d_max, seed=seed),
                final_reward=e_inter,
                z_table=z_table,
            )

    print(sars_list[:15])


if __name__ == '__main__':
    main()
