import argparse
import collections
import glob
import json
import os
import re
import sys
from typing import List, Optional, Tuple, Dict, DefaultDict

import matplotlib.pyplot as plt
import pandas as pd

fig_width = 2.5
fig_height = 2.1

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot SAC training statistics')
    parser.add_argument('--path', help='path to results file or directory', required=True)
    parser.add_argument('--min_iter', help='minimum iteration', default=0, type=int, required=False)
    parser.add_argument('--max_iter', help='maximum iteration', default=sys.maxsize, type=int, required=False)
    return parser.parse_args()


def get_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]

    paths = glob.glob(os.path.join(path, '*.txt'))

    if len(paths) == 0:
        raise RuntimeError(f"Cannot find results in '{path}'")

    return paths


name_re = re.compile(r'(?P<name>.+)_run-(?P<seed>\d+).txt')


def parse_path(path: str) -> Tuple[str, int]:
    match = name_re.match(os.path.basename(path))
    if not match:
        raise RuntimeError(f'Cannot parse {path}')

    return match.group('name'), int(match.group('seed'))


def parse_results(path: str, kind: Optional[str] = None) -> List[dict]:
    results = []
    with open(path, mode='r', encoding='ascii') as f:
        for line in f:
            d = json.loads(line)
            if (kind is not None) and (d['kind'] == kind):
                results.append(d)

    return results


def generate_df(dicts: List[Dict], seed: int) -> pd.DataFrame:
    df = pd.DataFrame(dicts)
    df['seed'] = seed
    return df


def generate_opt_df(dicts: List[Dict], seed: int) -> pd.DataFrame:
    sub_dfs = []
    for item in dicts:
        df = pd.DataFrame(item['progress'])
        for k in ['kind', 'iteration']:
            df[k] = item[k]
        sub_dfs.append(df)

    df = pd.concat(sub_dfs)
    df['seed'] = seed
    return df.reset_index(drop=True)


def plot_rollouts(ax: plt.Axes, df: pd.DataFrame, min_iter: int, max_iter: int, name: str) -> None:
    df = df[(min_iter <= df['iteration']) & (df['iteration'] <= max_iter)]
    df = df.groupby(['iteration']).agg(['mean', 'std']).reset_index()

    ax.plot(df['iteration'], df['return']['mean'], zorder=1, label=name)


def plot_optimization(ax: plt.Axes, df: pd.DataFrame, min_iter: int, max_iter: int) -> None:
    df = df[(min_iter <= df['iteration']) & (df['iteration'] <= max_iter)]
    df = df.groupby(['iteration', 'epoch']).agg(['mean', 'std']).reset_index()

    for k in ['loss_q', 'surrogate_loss_pi']:
        ax.plot(df.index, df[k]['mean'], label=k)


def group_by_name(tuples: List[Tuple[str, int, str]]) -> DefaultDict[str, List[Tuple[int, str]]]:
    d = collections.defaultdict(list)
    for t in tuples:
        d[t[0]].append((t[1], t[2]))
    return d


def analyse_and_plot_data(name: str, tuples: List[Tuple[int, str]], min_iter: int, max_iter: int) -> None:
    print(f"Parsing paths for '{name}': " + str([path for counter, path in tuples]))

    train_df = pd.concat([generate_df(parse_results(path, 'train'), seed=seed) for (seed, path) in tuples])
    eval_df = pd.concat([generate_df(parse_results(path, 'eval'), seed=seed) for (seed, path) in tuples])
    opt_df = pd.concat([generate_opt_df(parse_results(path, 'opt'), seed=seed) for (seed, path) in tuples])

    # Plot
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(2 * fig_width, fig_height), constrained_layout=True)

    # Rollouts
    plot_rollouts(axes[0], train_df, min_iter=min_iter, max_iter=max_iter, name='train')
    plot_rollouts(axes[0], eval_df, min_iter=min_iter, max_iter=max_iter, name='eval')

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Return')
    axes[0].legend()

    # Optimization
    plot_optimization(axes[1], opt_df, min_iter=min_iter, max_iter=max_iter)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    fig.savefig(f'training_{name}.pdf')
    plt.close(fig)


def main():
    args = parse_args()
    all_tuples = [(*parse_path(path), path) for path in get_paths(args.path)]
    grouped_tuples = group_by_name(all_tuples)  # type: ignore

    for name, tuples in grouped_tuples.items():
        analyse_and_plot_data(name=name, tuples=tuples, min_iter=args.min_iter, max_iter=args.max_iter)


if __name__ == '__main__':
    main()
