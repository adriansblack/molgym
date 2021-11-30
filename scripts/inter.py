import argparse
import logging
from typing import Dict

import ase
from e3nn import o3

from molgym import tools, data, rl


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--z_energies', help='atomic energy of elements (e.g.,: z1:e1, z2:e2)', type=str, required=True)
    return parser


def parse_z_energies(z_energies: str) -> Dict[int, float]:
    d = {0: 0.0}
    for item in z_energies.split(','):
        z_str, e_str = item.split(':')
        d[int(z_str)] = float(e_str)

    return d


def get_interaction_energy(config: data.Configuration, z_energies: Dict[int, float]) -> float:
    assert config.energy is not None
    return config.energy - sum(z_energies[z] for z in config.atomic_numbers)


def main() -> None:
    parser = tools.build_default_arg_parser()
    add_arguments(parser)
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

    policy = rl.Policy(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        network_width=args.network_width,
        num_gaussians=args.num_gaussians,
        min_max_distance=(args.d_min, args.d_max),
        gamma=args.gamma,
    )
    policy.to(device)

    reward_fn = rl.SparseInteractionReward()
    initial_state = data.get_empty_canvas_state(ase.Atoms('H2O'), z_table)
    logging.info('Initial state: ' + str(initial_state))
    buffer = rl.rollout(
        policy=policy,
        envs=rl.EnvironmentCollection(
            [rl.DiscreteMolecularEnvironment(reward_fn, initial_state, z_table) for _ in range(3)]),
        num_steps=12,
        num_episodes=None,
        d_max=args.d_max,
        batch_size=args.batch_size,
        training=True,
        device=device,
    )

    for sars in buffer:
        print(sars)
        if sars.done:
            print('----------')

    logging.info('Done')


if __name__ == '__main__':
    main()
