import argparse


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line tool of MolGen3D')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--checkpoints_dir', help='directory for checkpoint files', type=str, default='checkpoints')

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
    parser.add_argument('--num_interactions', help='number of interactions', type=int, default=3)
    parser.add_argument('--hidden_irreps',
                        help='irreps for hidden node states',
                        type=str,
                        default='16x0e + 16x1o + 16x2e + 16x3o')
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)
    parser.add_argument('--num_gaussians', help='number of Gaussians in GMM', type=int, default=3)

    # Actions
    parser.add_argument('--d_min', help='minimum distance (in Ang)', type=float, default=0.9)
    parser.add_argument('--d_max', help='maximum distance (in Ang)', type=float, default=1.8)

    # Path generation
    parser.add_argument('--num_paths_per_atom', help='number of paths per configuration per atom', type=int, default=1)

    return parser
