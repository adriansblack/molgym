import numpy as np
import plotly.graph_objects as go
import torch
from e3nn import o3

from molgym.spherical_distrs import SO3Distribution
from molgym.tools import set_seeds


def get_ring_samples(beta: float, num: int) -> torch.Tensor:
    alphas = torch.linspace(0, 2 * np.pi, steps=num + 1)[:-1]
    betas = torch.tensor([beta])
    alpha, beta = torch.meshgrid(alphas, betas)
    return o3.angles_to_xyz(alpha, beta).reshape(-1, 3)


def s2_grid(num=80) -> torch.Tensor:
    betas = torch.linspace(0, np.pi, num)
    alphas = torch.linspace(0, 2 * np.pi, num)
    beta, alpha = torch.meshgrid(betas, alphas)
    return o3.angles_to_xyz(alpha, beta)


def loss_fn(spherical: SO3Distribution, data: torch.Tensor) -> torch.Tensor:
    return -1 * torch.sum(spherical.log_prob(data))  # sum over S and B (B=1)


def optimize_parameters(coeffs: torch.Tensor, lmax: int, gamma: float, data: torch.Tensor, max_num_epochs=5000):
    optimizer = torch.optim.Adam(params=[coeffs], lr=1E-2, amsgrad=True)

    prev_loss = np.inf
    for i in range(max_num_epochs):
        optimizer.zero_grad()
        spherical = SO3Distribution(a_lms=coeffs, lmax=lmax, gamma=gamma)
        loss = loss_fn(spherical, data)
        loss.backward()
        optimizer.step()

        loss = loss.detach().numpy()
        if np.abs(loss - prev_loss) > 1e-6:
            prev_loss = loss
        else:
            print(f'Converged after {i+1} steps')
            break

    spherical = SO3Distribution(a_lms=coeffs, lmax=lmax, gamma=gamma)
    print(f'Final loss {loss_fn(spherical, data).item():.3f}')


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


def plot(grid, values, points_list) -> None:
    axis = dict(
        # showbackground=False,
        # showticklabels=False,
        # showgrid=True,
        # zeroline=False,
        # title='',
        nticks=3, )

    layout = dict(
        width=480,
        height=480,
        scene=dict(
            xaxis=dict(**axis, range=[-1.25, 1.25]),
            yaxis=dict(**axis, range=[-1.25, 1.25]),
            zaxis=dict(**axis, range=[-1.25, 1.25]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-5, z=5),
                projection=dict(type='orthographic'),
            ),
            dragmode='orbit',
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    shifted_values = (values - torch.min(values))
    radius = 0.6 + 0.4 * shifted_values / (torch.max(shifted_values) + 1e-4)

    trace = dict(x=radius * grid[..., 0], y=radius * grid[..., 1], z=radius * grid[..., 2], surfacecolor=values)
    color_max = trace['surfacecolor'].abs().max().item()
    color_min = trace['surfacecolor'].abs().min().item()
    data = [go.Surface(**trace, cmin=-color_min, cmax=color_max, colorscale='Viridis')]

    # Scatter
    scatter_r = 1.05
    color_iter = iter(colors)
    for points in points_list:
        scatter = go.Scatter3d(
            x=scatter_r * points[..., 0].flatten(),
            y=scatter_r * points[..., 1].flatten(),
            z=scatter_r * points[..., 2].flatten(),
            mode='markers',
            marker=dict(
                size=2,
                color=next(color_iter),
            ),
        )
        data.append(scatter)

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def main():
    set_seeds(1)

    num_datapoints = 1
    num_samples = 25
    lmax = 3
    gamma = 25

    # Note: sample, batch, event (S, B, E)
    data1 = torch.cat([
        get_ring_samples(beta=np.pi / 3, num=num_datapoints),
        get_ring_samples(beta=2 * np.pi / 3, num=num_datapoints)
    ],
                      dim=0)
    data2 = get_ring_samples(beta=np.pi / 2, num=2 * num_datapoints)
    data3 = get_ring_samples(beta=np.pi / 3, num=2 * num_datapoints)
    data_list = [data1, data2, data3]
    data = torch.stack(data_list).transpose(0, 1)
    print('Samples:', data.shape)  # [S, B, E]

    # Optimize coefficients
    coeffs = torch.randn(data.shape[1], (lmax + 1)**2, requires_grad=True)
    optimize_parameters(coeffs, lmax=lmax, gamma=gamma, data=data)

    spherical = SO3Distribution(a_lms=coeffs, lmax=lmax, gamma=gamma)
    print(spherical)

    # Compute logp's on mesh and sample
    grid = s2_grid()
    values_list = torch.exp(spherical.log_prob(grid.unsqueeze(-2)).detach()).split(1, dim=-1)  # [S..., B]
    samples_list = spherical.sample(torch.Size((num_samples, ))).split(1, dim=-2)  # [S, B, E]

    # Visualize
    for values, samples, data in zip(values_list, samples_list, data_list):
        plot(
            grid=grid,
            values=values.squeeze(-1),
            points_list=[data, samples],
        )


if __name__ == '__main__':
    main()
