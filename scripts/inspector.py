import argparse
from typing import Sequence, Tuple

import ase.data
import ase.io
import numpy as np
import plotly.graph_objects as go
import torch
import torch_geometric
from e3nn import o3
from plotly.subplots import make_subplots

from molgym import data, tools, distributions
from molgym.tools import to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Agent Inspector')

    parser.add_argument('--model', help='path to model', type=str, required=True)
    parser.add_argument('--zs', help='atomic numbers in table', type=str, required=True)
    parser.add_argument('--xyz', help='path to trajectory (.xyz)', type=str, required=True)

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
    focus: int = None,
) -> None:
    probs = to_numpy(distr.distr.probs[0])
    indices = np.arange(len(probs))

    fig.add_trace(go.Bar(x=indices, y=probs, name='p(f)'), row=row, col=col)

    fig.update_xaxes(title_text='f', tickmode='array', tickvals=indices, ticktext=indices, row=row, col=col)
    fig.update_yaxes(title_text='p(f)', row=row, col=col)


def plot_element_distribution(
    fig,
    row: int,
    col: int,
    distr: torch.distributions.Categorical,
    labels: Sequence[str],
    element: int = None,
) -> None:
    probs = to_numpy(distr.probs[0])
    indices = np.arange(len(probs))

    assert len(labels) == len(probs)

    fig.add_trace(go.Bar(x=indices, y=probs, name='p(f)'), row=row, col=col)

    fig.update_xaxes(title_text='e', tickmode='array', tickvals=indices, ticktext=labels, row=row, col=col)
    fig.update_yaxes(title_text='p(e)', row=row, col=col)


def plot_distance_distribution(
    fig,
    row: int,
    col: int,
    distr: distributions.GaussianMixtureModel,
    distance_range: Tuple[float, float],
    distance: float = None,
) -> None:
    plot_range = (distance_range[0] - 0.15, distance_range[1] + 0.15)
    ds = np.linspace(start=plot_range[0], stop=plot_range[1], num=128)
    probs = to_numpy(torch.exp(distr.log_prob(torch.tensor(ds))))

    fig.add_trace(go.Scatter(x=ds, y=probs, mode='lines', name='p(d)'), row=row, col=col)

    fig.update_xaxes(title_text='d', row=row, col=col)
    fig.update_yaxes(title_text='p(d)', row=row, col=col)


grid = s2_grid()


def plot_orientation_distribution(fig, row, col, distr: distributions.SO3Distribution) -> None:
    num_samples = 32
    values = torch.exp(distr.log_prob(grid.unsqueeze(-2)).detach())[..., 0]  # [S..., B]
    samples = distr.sample(torch.Size((num_samples,))).split(1, dim=-2)[0]  # [S, B, E]

    shifted_values = (values - torch.min(values))
    radius = 0.6 + 0.4 * shifted_values / (torch.max(shifted_values) + 1e-4)

    surface = go.Surface(
        x=radius * grid[..., 0],
        y=radius * grid[..., 1],
        z=radius * grid[..., 2],
        surfacecolor=values,
        colorscale='Viridis',
        cmax=values.max().item(),
        cmin=values.min().item(),
        colorbar=dict(
            title='p(x)',
            titleside='right',
            len=0.3,
            y=0.2,
        ),
    )
    fig.add_trace(surface, row=row, col=col)

    # Scatter
    scatter_r = 1.05
    color_iter = iter(colors)
    scatter = go.Scatter3d(
        x=scatter_r * samples[..., 0].flatten(),
        y=scatter_r * samples[..., 1].flatten(),
        z=scatter_r * samples[..., 2].flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color=next(color_iter),
        ),
        name='Spherical Samples',
    )
    fig.add_trace(scatter, row=row, col=col)

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

    model = torch.load(f=args.model, map_location=device)

    atoms = ase.io.read(args.xyz, format='extxyz', index=0)
    z_table = data.AtomicNumberTable([int(z) for z in args.zs.split(',')])
    sars_list = data.generate_sparse_reward_trajectory(atoms, z_table, final_reward=0.0)

    geometric_data = [
        data.build_state_action_data(state=item.state, cutoff=args.d_max, action=item.action)
        for item in sars_list
    ]

    data_loader = torch_geometric.loader.DataLoader(
        dataset=geometric_data,
        # batch_size=len(geometric_data),
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    iterator = iter(data_loader)
    next(iterator)
    next(iterator)
    next(iterator)
    next(iterator)
    batch = next(iterator)
    batch = batch.to(device)

    output, aux = model(batch)
    print(output, aux)

    # Visualize
    fig = make_subplots(
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
        subplot_titles=('Focus f', 'Element e', 'Distance d', 'Orientation x'),
    )
    fig.update_layout(width=1200, height=1000)

    plot_focus_distribution(fig, row=1, col=1, distr=aux['distrs'][0])

    symbols = [ase.data.chemical_symbols[z] for z in z_table.zs]
    plot_element_distribution(fig, row=1, col=2, distr=aux['distrs'][1], labels=symbols)

    distance_range = (float(args.d_min), float(args.d_max))
    plot_distance_distribution(fig, row=2, col=1, distance_range=distance_range, distr=aux['distrs'][2])

    plot_orientation_distribution(fig, row=2, col=2, distr=aux['distrs'][3])

    fig.show()


if __name__ == '__main__':
    main()
