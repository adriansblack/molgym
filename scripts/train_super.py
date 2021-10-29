import argparse
import logging
from typing import Dict

import numpy as np
import torch
import torch_geometric
from e3nn import o3

from molgym import tools, data, modules
from molgym.data import graph_tools


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

        for seed in range(args.num_paths_per_config):
            sequence = graph_tools.breadth_first_rollout(graph, seed=seed)
            sars_list += data.generate_sparse_reward_trajectory(
                atoms=graph_tools.select_atoms(atoms, sequence),
                final_reward=e_inter,
                z_table=z_table,
            )

    geometric_data = [
        data.build_state_action_data(state=item.state, action=item.action, z_table=z_table, cutoff=args.d_max)
        for item in sars_list
    ]

    train_data, valid_data = tools.random_train_valid_split(geometric_data, valid_fraction=0.1, seed=1)
    logging.info(f'Training data: {len(train_data)}, valid data: {len(valid_data)}')

    train_loader, valid_loader = [
        torch_geometric.loader.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
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
        gamma=10.0,
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
        patience=np.inf,
        device=device,
    )


if __name__ == '__main__':
    main()
