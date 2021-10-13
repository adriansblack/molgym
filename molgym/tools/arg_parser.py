import argparse


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line tool of MolGen3D')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--downloads_dir', help='directory for downloads', type=str, default='downloads')
    parser.add_argument('--checkpoints_dir', help='directory for checkpoint files', type=str, default='checkpoints')

    # Device and logging
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')

    return parser
