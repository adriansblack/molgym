import argparse
import logging
import os
from typing import Any, Dict, List

import ase
import ase.io
import ase.data
import numpy as np
import torch
from e3nn import o3

from molgym import tools, data, rl


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--zs', help='atomic numbers (e.g.: 1,6,7,8)', type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--bag', help='chemical formula of initial state (e.g.: H2O)', type=str, required=False)
    group.add_argument('--initial_state', help='path to XYZ file', type=str, required=False)
    return parser


def create_trajectories(
    terminal_state: data.State,
    start_index: int,
    final_reward: float,
    cutoff: float,
    num_paths_per_atom: float,
    seed: int,
) -> List[data.Trajectory]:
    graph = data.graph_tools.generate_topology(terminal_state.positions, cutoff_distance=cutoff)
    num_paths = int(num_paths_per_atom * len(terminal_state.elements))

    taus: List[data.Trajectory] = []
    for i in range(num_paths):
        sequence = data.graph_tools.breadth_first_rollout(graph, seed=seed + i, visited=list(range(start_index)))
        tau = data.generate_sparse_reward_trajectory(
            terminal_state=data.State(elements=terminal_state.elements[sequence],
                                      positions=terminal_state.positions[sequence],
                                      bag=terminal_state.bag),
            final_reward=final_reward,
            start_index=start_index,
        )
        taus.append(tau)

    return taus


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
    logging.info(agent)
    target = rl.SACTarget(agent)
    logging.info(target)
    logging.info(f'Number of parameters: {sum(tools.count_parameters(m) for m in [agent, target])}')

    logging.info('Saving model')
    os.makedirs(name=args.checkpoint_dir, exist_ok=True)
    torch.save(agent.cpu(), os.path.join(args.checkpoint_dir, f'init_agent_{tag}.model'))

    # Optimizers
    optimizer = torch.optim.AdamW(
        params=agent.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    # Set up environment(s)
    reward_fn = rl.SparseInteractionReward()
    if args.initial_state:
        atoms = ase.io.read(args.initial_state, index=0, format='extxyz')
        initial_state = data.state_from_atoms(atoms, z_table=z_table)
    else:
        dummy_state = data.state_from_atoms(ase.Atoms(args.bag), z_table=z_table)
        initial_state = data.rewind_state(dummy_state, 0)

    logging.info(f'Initial state: canvas={len(initial_state.elements)} atom(s), bag={initial_state.bag}')
    envs = rl.EnvironmentCollection([
        rl.DiscreteMolecularEnvironment(reward_fn, initial_state, z_table, min_reward=args.min_reward)
        for _ in range(args.num_envs)
    ])

    # Checkpointing
    handler = tools.CheckpointHandler(directory=args.checkpoint_dir, tag=tag, keep=True)

    # Send models to device
    agent.to(device)
    target.to(device)

    highest_return = -np.inf
    dataset = []
    trajectory_lengths = []
    for i in range(args.num_iters):
        logging.info(f'Iteration {i}')

        # Collect data
        num_episodes = args.num_episodes_per_iter if i > 0 else args.num_initial_episodes
        logging.debug(f'Rollout with {num_episodes} episodes')
        new_trajectories = rl.rollout(
            agent=agent,
            envs=envs,
            num_steps=None,
            num_episodes=num_episodes,
            d_max=args.d_max,
            batch_size=args.batch_size,
            training=True,
            device=device,
        )

        # Analyze trajectories
        logging.debug('Analyzing trajectories')
        tau_info: Dict[str, Any] = data.analyze_trajectories(new_trajectories)
        tau_info['iteration'] = i
        tau_info['kind'] = 'train'
        logger.log(tau_info)

        # Create new trajectories
        generated_trajectories = []
        for new_trajectory in new_trajectories:
            # Check if episode was terminated by environment
            if new_trajectory[-1].reward <= args.min_reward:
                continue

            first_state = new_trajectory[0].state
            generated_trajectories += create_trajectories(
                terminal_state=new_trajectory[-1].next_state,
                # make sure the first atom is not a dummy atom
                start_index=len(first_state.elements) if first_state.elements[0] != 0 else 0,
                final_reward=new_trajectory[-1].reward,
                cutoff=args.r_max,
                num_paths_per_atom=args.num_paths_per_atom,
                seed=args.seed,
            )
        new_trajectories += generated_trajectories

        # Update buffer
        trajectory_lengths += [len(tau) for tau in new_trajectories]
        if args.max_num_episodes is not None:
            trajectory_lengths = trajectory_lengths[-args.max_num_episodes:]
        dataset += [data.process_sars(sars=sars, cutoff=args.d_max) for tau in new_trajectories for sars in tau]
        dataset = dataset[-sum(trajectory_lengths):]

        logging.debug(f'Preparing dataloader with {len(dataset)} steps')
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
            'kind': 'opt',
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
            tau_eval['kind'] = 'eval'
            logger.log(tau_eval)
            logging.info(f'eval_return={tau_eval["return"]:.3f}')

            terminal_atoms = [
                data.state_to_atoms(tau[-1].next_state, z_table, info={'reward': tau[-1].reward})
                for tau in eval_trajectories
            ]
            ase.io.write(os.path.join(args.log_dir, f'terminals_{tag}_{i}.xyz'), images=terminal_atoms, format='extxyz')

            if tau_eval['return'] > highest_return:
                highest_return = tau_eval['return']
                handler.save(tools.CheckpointState(agent, optimizer), counter=i)

    logging.info('Saving model')
    os.makedirs(name=args.checkpoint_dir, exist_ok=True)
    torch.save(agent.cpu(), os.path.join(args.checkpoint_dir, f'agent_{tag}.model'))

    logging.info('Done')


if __name__ == '__main__':
    main()
