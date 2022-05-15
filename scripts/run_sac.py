import argparse
import dataclasses
import logging
import os
from typing import Any, Dict, List

import ase
import ase.data
import ase.io
import networkx as nx
import numpy as np
import torch
from e3nn import o3

from molgym import tools, data, rl


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--symbols', help='symbols (e.g.: XHCO, X is required)', type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--bag', help='chemical formula of initial state (e.g.: H2O)', type=str, required=False)
    group.add_argument('--initial_state', help='path to XYZ file', type=str, required=False)
    group.add_argument('--atomcosts', help='per atom costs', type=str, required=False)
    return parser


@dataclasses.dataclass
class EndPoint:
    state: data.State
    start_index: int
    reward: float


def create_trajectories(
    end_point: EndPoint,
    cutoff: float,
    num_paths: int,
    seed: int,
) -> List[data.Trajectory]:
    graph = data.graph_tools.generate_topology(end_point.state.positions, cutoff_distance=cutoff)
    if not nx.is_connected(graph):
        return []

    taus: List[data.Trajectory] = []
    for i in range(num_paths):
        sequence = data.graph_tools.breadth_first_rollout(graph,
                                                          seed=seed + i,
                                                          visited=list(range(end_point.start_index)))
        tau = data.generate_sparse_reward_trajectory(
            terminal_state=data.State(elements=end_point.state.elements[sequence],
                                      positions=end_point.state.positions[sequence],
                                      bag=end_point.state.bag),
            final_reward=end_point.reward,
            start_index=end_point.start_index,
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
    s_table = data.SymbolTable(args.symbols)
    logging.info(s_table)

    # Create modules
    agent = rl.SACAgent(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        num_elements=len(s_table),
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
        initial_state = data.state_from_atoms(atoms, s_table=s_table)
    else:
        dummy_state = data.state_from_atoms(ase.Atoms(args.bag), s_table=s_table)
        initial_state = data.rewind_state(dummy_state, 0)

    logging.info(f'Initial state: canvas={len(initial_state.elements)} atom(s), bag={initial_state.bag}')
    envs = rl.EnvironmentCollection([
        rl.DiscreteMolecularEnvironment(reward_fn, initial_state, s_table, min_reward=args.min_reward)
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
    total_num_episodes = 0
    for i in range(args.num_iters):
        logging.info(f'Iteration {i}')

        # Collect data
        num_episodes = args.num_episodes_per_iter if i > 0 else args.num_initial_episodes
        logging.debug(f'Rollout with {num_episodes} episodes')
        new_taus = rl.rollout(
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
        tau_info: Dict[str, Any] = data.analyze_trajectories(new_taus)
        tau_info['iteration'] = i
        tau_info['total_num_episodes'] = total_num_episodes
        tau_info['kind'] = 'rollout'
        logger.log(tau_info)

        # Structure optimization
        if args.max_opt_iters > 0:
            logging.debug('Optimizing trajectories')
            end_points = []
            for tau in new_taus:
                terminal_state = tau[-1].next_state
                configs, _success = rl.optimize_structure(
                    reward_fn=reward_fn,
                    symbols=[s_table.element_to_symbol(e) for e in terminal_state.elements],
                    positions=terminal_state.positions,
                    max_iter=args.max_opt_iters,
                )
                end_points += [
                    EndPoint(
                        state=data.State(
                            elements=np.array([s_table.symbol_to_element(s) for s in config.symbols], dtype=int),
                            positions=config.positions,
                            bag=terminal_state.bag,
                        ),
                        start_index=len(tau[0].state.elements) if tau[0].state.elements[0] != 0 else 0,
                        reward=config.reward,
                    ) for config in configs
                ]
                total_num_episodes += len(configs)
        else:
            end_points = [
                EndPoint(
                    state=tau[-1].next_state,
                    start_index=len(tau[0].state.elements) if tau[0].state.elements[0] != 0 else 0,
                    reward=tau[-1].reward,
                ) for tau in new_taus
            ]
            total_num_episodes += len(new_taus)

        # Generate new trajectories from EndPoints
        generated_trajectories = []
        for end_point in end_points:
            # Check if episode was terminated by environment
            if end_point.reward <= args.min_reward:
                continue

            generated_trajectories += create_trajectories(
                end_point=end_point,
                cutoff=args.r_max,
                num_paths=max(int(args.num_paths_per_atom * len(end_point.state.elements)), 1),
                seed=args.seed,
            )
        new_taus = generated_trajectories

        logging.debug('Analyzing generated trajectories')
        tau_info: Dict[str, Any] = data.analyze_trajectories(new_taus)
        tau_info['iteration'] = i
        tau_info['total_num_episodes'] = total_num_episodes
        tau_info['kind'] = 'train'
        logger.log(tau_info)

        # Update buffer
        trajectory_lengths += [len(tau) for tau in new_taus]
        if args.max_num_episodes is not None:
            trajectory_lengths = trajectory_lengths[-args.max_num_episodes:]
        dataset += [data.process_sars(sars=sars, cutoff=args.d_max) for tau in new_taus for sars in tau]
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
            'total_num_episodes': total_num_episodes,
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
            tau_eval['total_num_episodes'] = total_num_episodes
            tau_eval['kind'] = 'eval'
            logger.log(tau_eval)
            logging.info(f'eval_return={tau_eval["return"]:.3f}')

            terminal_atoms = [
                data.state_to_atoms(tau[-1].next_state, s_table, info={'reward': tau[-1].reward})
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
