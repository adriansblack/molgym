import argparse
import logging
from typing import List

import ase
from e3nn import o3

from molgym import tools, data, rl


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--zs', help='atomic numbers (e.g.,: 1,6,7,8)', type=str, required=True)
    return parser


def prepare_zs(zs_str: str) -> List[int]:
    z_ints = sorted(int(i) for i in zs_str.split(','))
    assert z_ints[0] >= 0

    # First Z has to be 0
    if z_ints[0] != 0:
        z_ints.insert(0, 0)

    return z_ints


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
    zs = prepare_zs(args.zs)
    z_table = data.AtomicNumberTable(zs)
    logging.info(z_table)

    # Create modules
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
    logging.info(policy)
    logging.info(f'Number of parameters: {tools.count_parameters(policy)}')

    reward_fn = rl.SparseInteractionReward()
    initial_state = data.get_state_from_atoms(ase.Atoms('H2O'), index=0, z_table=z_table)
    logging.info('Initial state: ' + str(initial_state))

    num_iterations = 3
    trajectories: List[data.Trajectory] = []
    for _ in range(num_iterations):
        new_trajectories = rl.rollout(
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


if __name__ == '__main__':
    main()
