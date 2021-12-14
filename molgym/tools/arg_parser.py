import argparse


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line arguments for MolGym3')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', help='directory for checkpoint files', type=str, default='checkpoints')

    # Device and logging
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')
    parser.add_argument('--default_dtype',
                        help='set default dtype',
                        type=str,
                        choices=['float32', 'float64'],
                        default='float64')

    # Model
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    parser.add_argument('--num_radial_basis', help='number of radial basis functions', type=int, default=8)
    parser.add_argument('--num_cutoff_basis', help='number of basis functions for smooth cutoff', type=int, default=6)
    parser.add_argument('--max_ell', help=r'highest \ell of spherical harmonics', type=int, default=3)
    parser.add_argument('--beta', help='beta parameter of spherical distribution', required=False, default=30)
    parser.add_argument('--num_interactions', help='number of interactions', type=int, default=3)
    parser.add_argument('--hidden_irreps',
                        help='irreps for hidden node states',
                        type=str,
                        default='16x0e + 16x1o + 16x2e + 16x3o')
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)
    parser.add_argument('--num_gaussians', help='number of Gaussians in GMM', type=int, default=3)

    # Loss and optimization
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--lr', help='learning rate of optimizer', type=float, default=0.001)
    parser.add_argument('--lr_scheduler_gamma', help='gamma of learning rate scheduler', type=float, default=0.9993)
    parser.add_argument('--weight_decay', help='weight decay (L2 penalty)', type=float, default=5e-5)
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=10)
    parser.add_argument('--eval_interval', help='evaluate model every <n> epochs', type=int, default=5)
    parser.add_argument('--keep_checkpoints', help='keep all checkpoints', action='store_true', default=False)
    parser.add_argument('--restart_latest',
                        help='restart optimizer from latest checkpoint',
                        action='store_true',
                        default=False)

    # Actions
    parser.add_argument('--d_min', help='minimum distance (in Ang)', type=float, default=0.9)
    parser.add_argument('--d_max', help='maximum distance (in Ang)', type=float, default=1.8)

    # RL
    parser.add_argument('--num_rollouts', help='number of rollouts', type=int, default=16)
    parser.add_argument('--num_buffers', help='number of rollouts to keep in the buffer', type=int, default=8)
    parser.add_argument('--num_iters', help='maximum number of iterations', type=int, default=100)
    parser.add_argument('--num_envs', help='number of environment copies', type=int, default=4)
    parser.add_argument('--num_paths_per_atom',
                        help='number of paths per atom in configuration',
                        type=float,
                        default=0.5)
    parser.add_argument('--alpha', help='alpha term in maximum entropy RL', type=float, default=0.2)
    parser.add_argument('--polyak',
                        help='interpolation factor in polyak averaging for target networks',
                        type=float,
                        default=0.995)

    return parser
