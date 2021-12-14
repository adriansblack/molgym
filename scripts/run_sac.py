import argparse
import logging
import os
from typing import List, Any, Dict

import ase
import ase.io
import numpy as np
import torch
from e3nn import o3

from molgym import tools, data, rl


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--zs', help='atomic numbers (e.g.: 1,6,7,8)', type=str, required=True)
    parser.add_argument('--bag', help='chemical formula of initial state (e.g.: H2O)', type=str, required=True)
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
    logger = tools.MetricsLogger(directory=args.log_dir, tag=tag)

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
    optimizer = torch.optim.AdamW(
        params=agent.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    # Set up environment(s)
    reward_fn = rl.SparseInteractionReward()
    initial_state = data.get_state_from_atoms(ase.Atoms(args.bag), index=0, z_table=z_table)
    logging.info('Initial state: ' + str(initial_state))
    envs = rl.EnvironmentCollection(
        [rl.DiscreteMolecularEnvironment(reward_fn, initial_state, z_table) for _ in range(args.num_envs)])

    highest_return = -np.inf
    trajectories: List[data.Trajectory] = []
    for i in range(args.num_iters):
        # Collect data
        logging.debug(f'Rollout with {args.num_rollouts} episodes')
        new_trajectories = rl.rollout(
            agent=agent,
            envs=envs,
            num_steps=None,
            num_episodes=args.num_rollouts,
            d_max=args.d_max,
            batch_size=args.batch_size,
            training=True,
            device=device,
        )

        # Analyze trajectories
        logging.debug('Analyzing trajectories')
        tau_info: Dict[str, Any] = data.analyze_trajectories(new_trajectories)
        tau_info['iteration'] = i
        tau_info['kind'] = 'train_rollout'
        logger.log(tau_info)

        # Update buffer
        trajectories += new_trajectories
        trajectories = trajectories[-args.num_buffers * args.num_rollouts:]

        # Prepare data
        logging.debug(f'Preparing {len(trajectories)} trajectories')
        dataset = [data.process_sars(sars=sars, cutoff=args.d_max) for tau in trajectories for sars in tau]
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=min(args.batch_size, len(dataset)),
            shuffle=True,
            drop_last=True,
        )

        # Train
        info = rl.train_sac(
            ac=agent,
            ac_target=target,
            optimizer=optimizer,
            data_loader=data_loader,
            alpha=args.alpha,
            polyak=args.polyak,
            cutoff=args.d_max,
            device=device,
            num_epochs=args.num_epochs,
        )
        train_info = {
            'progress': info,
            'kind': 'train',
            'iteration': i,
        }
        logger.log(train_info)

        # Evaluate
        if i % args.eval_interval == 0:
            logging.debug('Evaluation rollout')
            eval_trajectories = rl.rollout(
                agent=agent,
                envs=envs,
                num_steps=None,
                num_episodes=1,
                d_max=args.d_max,
                batch_size=args.batch_size,
                training=False,
                device=device,
            )
            tau_eval: Dict[str, Any] = data.analyze_trajectories(eval_trajectories)
            tau_eval['iteration'] = i
            tau_eval['kind'] = 'eval_rollout'
            logger.log(tau_eval)
            logging.info(f'Evaluation return: {tau_eval["return"]:.3f}')

            terminal_atoms = [data.state_to_atoms(tau[-1].next_state, z_table) for tau in eval_trajectories]
            ase.io.write(os.path.join(args.log_dir, f'terminals_{i}.xyz'), images=terminal_atoms, format='extxyz')

            if tau_eval['return'] > highest_return:
                highest_return = tau_eval['return']
                os.makedirs(name=args.checkpoint_dir, exist_ok=True)
                agent_cpu = agent.to(torch.device('cpu'))
                torch.save(agent_cpu, os.path.join(args.checkpoint_dir, f'agent_{i}.model'))


if __name__ == '__main__':
    main()
