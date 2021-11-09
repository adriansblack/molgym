import argparse
import logging
import os
from typing import Dict, List

import ase.io
import torch
import torch_geometric
from e3nn import o3

from molgym import tools, data, modules
from molgym.data import graph_tools, SARS


def add_supervised(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--xyz', help='path to XYZ file', type=str, required=True)
    parser.add_argument('--max_num_configs', help='maximum number of samples', type=int, required=False, default=None)
    parser.add_argument('--z_energies', help='atomic energy of elements (e.g.,: z1:e1, z2:e2)', type=str, required=True)
    parser.add_argument('--num_sampled_trajectories',
                        help='number of sampled trajectories',
                        type=int,
                        required=False,
                        default=10)
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


def sample_trajectories(
    policy: modules.Policy,
    initial_state: data.DiscreteBagState,
    cutoff: float,
    count: int,
    device: torch.device,
) -> List[data.DiscreteBagState]:

    terminal_states = []

    for i in range(count):
        state = initial_state

        while True:
            loader = torch_geometric.loader.DataLoader(
                dataset=[data.build_state_action_data(state, cutoff=cutoff, action=None)],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )

            batch = next(iter(loader))
            batch = batch.to(device)
            response, aux = policy(batch)
            actions = data.build_actions(response)

            assert len(actions) == 1
            state = data.propagate_discrete_bag_state(state, actions[0])

            if data.bag_is_empty(state.bag):
                terminal_states.append(state)
                break

    return terminal_states


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
    atoms_list = data.load_xyz(args.xyz)[:args.max_num_configs]

    # Generate SARS list
    sars_list: List[SARS] = []
    seed = 0
    for atoms in atoms_list:
        e_inter = get_interaction_energy(config=data.config_from_atoms(atoms), z_energies=z_energies)
        graph = graph_tools.generate_topology(atoms, cutoff_distance=args.d_max)

        num_paths = max(int(args.num_paths_per_atom * len(atoms)), 1)
        for _ in range(num_paths):
            sequence = graph_tools.breadth_first_rollout(graph, seed=seed)
            sars_list += data.generate_sparse_reward_trajectory(
                atoms=graph_tools.select_atoms(atoms, sequence),
                final_reward=e_inter,
                z_table=z_table,
            )
            seed += 1

    geometric_data = [
        data.build_state_action_data(state=item.state, cutoff=args.d_max, action=item.action) for item in sars_list
    ]

    train_data, valid_data = tools.random_train_valid_split(geometric_data, valid_fraction=0.1, seed=1)
    logging.info(f'Training data: {len(train_data)}, valid data: {len(valid_data)}')

    train_loader, valid_loader = [
        torch_geometric.loader.DataLoader(
            dataset=dataset,
            batch_size=min(args.batch_size, len(dataset)),
            shuffle=True,
            drop_last=True,
        ) for dataset in (train_data, valid_data)
    ]

    policy = modules.Policy(
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

    optimizer = torch.optim.AdamW(
        params=policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_scheduler_gamma)
    checkpoint_handler = tools.CheckpointHandler(directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints)
    logger = tools.ProgressLogger(directory=args.log_dir, tag=tag + '_train')

    start_epoch = 0
    if args.restart_latest:
        start_epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(policy, optimizer, lr_scheduler),
                                                     device=device)

    logging.info(policy)
    logging.info(f'Number of parameters: {tools.count_parameters(policy)}')
    logging.info(f'Optimizer: {optimizer}')

    tools.train(
        model=policy,
        loss_fn=tools.neg_log_likelihood,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.max_num_epochs,
        device=device,
    )

    # Load policy
    epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(policy, optimizer, lr_scheduler), device=device)
    logging.info(f'Loaded model from epoch {epoch}')

    # Test policy
    initial_state = data.get_initial_state(atoms=atoms_list[0], z_table=z_table)
    terminals = sample_trajectories(
        policy,
        initial_state=initial_state,
        cutoff=args.d_max,
        count=args.num_sampled_trajectories,
        device=device,
    )
    terminal_atoms = [data.state_to_atoms(terminal_state, z_table) for terminal_state in terminals]
    ase.io.write(os.path.join(args.log_dir, tag + '_terminals.xyz'), terminal_atoms, format='extxyz')

    # Save policy as model
    policy_path = os.path.join(args.checkpoints_dir, tag + '.model')
    logging.info(f'Saving policy to {policy_path}')
    policy_cpu = policy.to(torch.device('cpu'))
    torch.save(policy_cpu, policy_path)

    logging.info('Done')


if __name__ == '__main__':
    main()
