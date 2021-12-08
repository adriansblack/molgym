import argparse
import logging
from typing import List

import ase
import torch
from e3nn import o3

from molgym import tools, data, rl


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--zs', help='atomic numbers (e.g.: 1,6,7,8)', type=str, required=True)
    return parser


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

    # Create Z table
    z_table = data.AtomicNumberTable(tools.parse_zs(args.zs))
    logging.info(z_table)

    # Create modules
    agent = rl.SACAgent(
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
        beta=args.beta,
    )
    agent.to(device)
    logging.info(agent)
    target = rl.SACTarget(agent)
    target.to(device)
    logging.info(target)
    logging.info(f'Number of parameters: {sum(tools.count_parameters(m) for m in [agent, target])}')

    # Optimizers
    pi_optimizer = torch.optim.AdamW(
        params=[{
            'name': 'policy',
            'params': agent.policy.parameters(),
        }],
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    q_optimizer = torch.optim.AdamW(
        params=[{
            'name': 'q1',
            'params': agent.q1.parameters(),
        }, {
            'name': 'q2',
            'params': agent.q2.parameters(),
        }],
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    # Set up environment(s)
    reward_fn = rl.SparseInteractionReward()
    initial_state = data.get_state_from_atoms(ase.Atoms('H2O'), index=0, z_table=z_table)
    logging.info('Initial state: ' + str(initial_state))
    envs = rl.EnvironmentCollection(
        [rl.DiscreteMolecularEnvironment(reward_fn, initial_state, z_table) for _ in range(3)])

    num_iterations = 3
    trajectories: List[data.Trajectory] = []
    for _ in range(num_iterations):
        new_trajectories = rl.rollout(
            agent=agent,
            envs=envs,
            num_steps=12,
            num_episodes=None,
            d_max=args.d_max,
            batch_size=args.batch_size,
            training=True,
            device=device,
        )

        trajectories += new_trajectories

        data_loader = data.DataLoader(
            dataset=[data.process_sars(sars=sars, cutoff=args.d_max) for tau in trajectories for sars in tau],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        rl.train_sac(
            ac=agent,
            ac_target=target,
            q_optimizer=q_optimizer,
            pi_optimizer=pi_optimizer,
            data_loader=data_loader,
            gamma=args.gamma,
            alpha=args.alpha,
            polyak=args.polyak,
            cutoff=args.d_max,
            device=device,
        )

    print(trajectories)


if __name__ == '__main__':
    main()