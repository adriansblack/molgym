import argparse
from typing import Sequence, Tuple

import ase.data
import ase.io
import numpy as np
import plotly.subplots
import torch
from e3nn import o3

from molgym import data, tools, distributions
from molgym.tools import to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Agent Inspector')

    parser.add_argument('--model', help='path to model', type=str, required=True)
    parser.add_argument('--checkpoint', help='path to checkpoint', type=str, required=False)
    parser.add_argument('--zs', help='atomic numbers in table', type=str, required=True)
    parser.add_argument('--xyz', help='path to trajectory (.xyz)', type=str, required=True)
    parser.add_argument('--index', help='config in XYZ file', type=int, required=False, default=0)

    parser.add_argument('--d_min', help='minimum distance (in Ang)', type=float, default=0.9)
    parser.add_argument('--d_max', help='maximum distance (in Ang)', type=float, default=1.8)

    parser.add_argument('--seed', help='set random seed', type=int, required=False, default=0)
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--default_dtype',
                        help='set default dtype',
                        type=str,
                        choices=['float32', 'float64'],
                        default='float64')

    return parser.parse_args()


def s2_grid(num=80) -> torch.Tensor:
    betas = torch.linspace(0, np.pi, num)
    alphas = torch.linspace(0, 2 * np.pi, num)
    beta, alpha = torch.meshgrid(betas, alphas)
    return o3.angles_to_xyz(alpha, beta)


colors = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]


def plot_focus_distribution(
    fig,
    row: int,
    col: int,
    distr: distributions.GraphCategoricalDistribution,
    focus: int,
) -> None:
    probs = to_numpy(distr.distr.probs[0])
    indices = np.arange(len(probs))

    c = ['lightslategray'] * len(probs)
    c[focus] = colors[0]

    fig.add_bar(
        x=indices,
        y=probs,
        name='p(f)',
        marker_color=c,
        text=probs,
        texttemplate='%{text:.2f}',
        textposition='outside',
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text='f', tickmode='array', tickvals=indices, ticktext=indices, row=row, col=col)
    fig.update_yaxes(title_text='p(f)', row=row, col=col)


def plot_element_distribution(
    fig,
    row: int,
    col: int,
    distr: torch.distributions.Categorical,
    labels: Sequence[str],
    element: int,
) -> None:
    probs = to_numpy(distr.probs[0])
    indices = np.arange(len(probs))

    cs = ['lightslategray'] * len(probs)
    cs[element] = colors[1]

    assert len(labels) == len(probs)

    fig.add_bar(
        x=indices,
        y=probs,
        marker_color=cs,
        name='p(e)',
        text=probs,
        texttemplate='%{text:.2f}',
        textposition='outside',
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text='e', tickmode='array', tickvals=indices, ticktext=labels, row=row, col=col)
    fig.update_yaxes(title_text='p(e)', row=row, col=col)


def plot_distance_distribution(
    fig,
    row: int,
    col: int,
    distr: distributions.GaussianMixtureModel,
    distance_range: Tuple[float, float],
    distance: float,
) -> None:
    plot_range = (distance_range[0] - 0.15, distance_range[1] + 0.15)
    ds = np.linspace(start=plot_range[0], stop=plot_range[1], num=128)
    probs = to_numpy(torch.exp(distr.log_prob(torch.tensor(ds))))

    fig.add_scatter(x=ds, y=probs, mode='lines', line_color='lightslategrey', name='p(d)', row=row, col=col)
    fig.add_vline(x=distance, line_color=colors[3], row=row, col=col, name='d')

    fig.update_xaxes(title_text='d', row=row, col=col)
    fig.update_yaxes(title_text='p(d)', row=row, col=col)


grid = s2_grid()


def plot_orientation_distribution(fig, row, col, distr: distributions.SO3Distribution, orientation: np.ndarray) -> None:
    num_samples = 32
    values = torch.exp(distr.log_prob(grid.unsqueeze(-2)).detach())[..., 0]  # [S..., B]
    samples = distr.sample(torch.Size((num_samples, ))).split(1, dim=-2)[0]  # [S, B, E]

    shifted_values = (values - torch.min(values))
    radius = 0.6 + 0.4 * shifted_values / (torch.max(shifted_values) + 1e-4)

    fig.add_surface(
        x=radius * grid[..., 0],
        y=radius * grid[..., 1],
        z=radius * grid[..., 2],
        surfacecolor=values,
        colorscale='Viridis',
        cmax=max(values.max().item(), 0.0),
        cmin=max(values.min().item(), 0.0),
        colorbar=dict(
            title='p(x)',
            titleside='right',
            len=0.3,
            y=0.2,
        ),
        name='p(o)',
        row=row,
        col=col,
    )

    # Scatter
    scatter_r = 1.05
    fig.add_scatter3d(
        x=scatter_r * samples[..., 0].flatten(),
        y=scatter_r * samples[..., 1].flatten(),
        z=scatter_r * samples[..., 2].flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color='lightslategrey',
        ),
        name='Sample',
        row=row,
        col=col,
    )

    fig.add_scatter3d(
        x=scatter_r * orientation[0].flatten(),
        y=scatter_r * orientation[1].flatten(),
        z=scatter_r * orientation[2].flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color=colors[2],
        ),
        name='Orientation',
        row=row,
        col=col,
    )

    fig.update_layout(scene=dict(
        xaxis=dict(nticks=3, range=[-1.25, 1.25]),
        yaxis=dict(nticks=3, range=[-1.25, 1.25]),
        zaxis=dict(nticks=3, range=[-1.25, 1.25]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-5, z=5),
            projection=dict(type='orthographic'),
        ),
        dragmode='orbit',
    ))


def main():
    args = parse_args()
    tools.set_seeds(args.seed)
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    # Load model (and update parameters)
    model = torch.load(f=args.model, map_location=device)
    if args.checkpoint:
        checkpoint = torch.load(f=args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)  # type: ignore

    # Load data
    atoms = ase.io.read(args.xyz, format='extxyz', index=args.index)

    # Parse Z table
    z_table = data.AtomicNumberTable([int(z) for z in args.zs.split(',')])
    symbols = [ase.data.chemical_symbols[z] for z in z_table.zs]

    focuses = atoms.info.get('focuses', None)
    assert len(focuses) == len(atoms) if focuses is not None else True
    terminal_state = data.get_state_from_atoms(atoms, z_table)
    sars_list = data.generate_sparse_reward_trajectory(terminal_state, final_reward=0.0, focuses=focuses)

    data_loader = data.DataLoader(
        dataset=[data.process_sa(state=item.state, cutoff=args.d_max, action=item.action) for item in sars_list],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    for batch in data_loader:
        batch = tools.dict_to_device(batch, device)
        output, aux = model(batch['state'], action=batch['action'], training=False)
        action = output['action']
        distrs = aux['distrs']

        # Visualize
        fig = plotly.subplots.make_subplots(
            rows=2,
            cols=2,
            specs=[[{
                'type': 'xy'
            }, {
                'type': 'xy'
            }], [{
                'type': 'xy'
            }, {
                'type': 'scene'
            }]],
            subplot_titles=('Focus', 'Element', 'Distance', 'Orientation'),
        )
        fig.update_layout(width=1200, height=1000, showlegend=False)

        plot_focus_distribution(fig, row=1, col=1, distr=distrs[0], focus=action['focus'][0])
        plot_element_distribution(fig, row=1, col=2, distr=distrs[1], labels=symbols, element=action['element'][0])
        plot_distance_distribution(fig,
                                   row=2,
                                   col=1,
                                   distance_range=(float(args.d_min), float(args.d_max)),
                                   distr=distrs[2],
                                   distance=action['distance'][0])
        plot_orientation_distribution(fig, row=2, col=2, distr=distrs[3], orientation=action['orientation'][0])

        fig.show()


if __name__ == '__main__':
    main()
