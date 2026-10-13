"""Recreate the FL scaling, deployment-cost, and Zipf-comparison figures."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from federated_learning.config import ROOT
from federated_learning.results import quantiles_or_nan, read_run


DATASETS = {
    'synthesized_v2_1_1.csv': 'S1',
    'synthesized_v2_2_1.csv': 'S2',
    'synthesized_v2_1_1_1000.csv': 'S3',
    'synthesized_v2_2_1_1000.csv': 'S4',
}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(best, output):
    fig, axes = plt.subplots(1, 4, figsize=(20.8, 3.4), sharey=True)
    for ax, (dataset, label) in zip(axes, DATASETS.items()):
        clients = [c for c in (10, 50, 100) if (dataset, c, 'homogeneous') in best]
        stats = [best[(dataset, c, 'homogeneous')]['best'] for c in clients]
        means = np.array([s['mean'] for s in stats])
        stds = np.array([s['std'] for s in stats])
        ax.plot(clients, means, 'o--', color='tab:green', linewidth=2)
        ax.fill_between(clients, means - stds, means + stds, color='tab:green', alpha=0.25)
        ax.set(title=label, xlabel='Number of clients C', xscale='log')
        ax.set_xticks([10, 50, 100], labels=['10', '50', '100'])
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('MSE (lower is better)')
    save_figure(fig, output / 'min_mse_vs_clients_quantile.pdf')


def plot_heterogeneity(best, output):
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    x = np.arange(len(DATASETS))
    for offset, mode, color, hatch, label in (
        (-0.18, 'homogeneous', 'tab:orange', '///', 'Homogeneous (C=10, equal split)'),
        (0.18, 'heterogeneous', 'tab:green', 'xxx', 'Heterogeneous (C=10, Zipf-1)'),
    ):
        stats = [best.get((dataset, 10, mode), {}).get('best', {}) for dataset in DATASETS]
        means = [s.get('mean', np.nan) for s in stats]
        stds = [s.get('std', 0.0) for s in stats]
        ax.bar(x + offset, means, 0.36, yerr=stds, capsize=4, color=color,
               edgecolor='black', linewidth=0.6, hatch=hatch, label=label)
    ax.set_xticks(x, labels=list(DATASETS.values()))
    ax.set_ylabel('MSE (lower is better)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    save_figure(fig, output / 'hom_vs_het_bars_quantile.pdf')


def overhead_rows(runs):
    groups = defaultdict(list)
    for run in runs:
        cfg = run['config']
        if cfg['aquacast_partition_sizes'] is None:
            groups[(cfg['num_sensors'], cfg['num_clients'])].append(run)
    rows = []
    for (nodes, clients), group in sorted(groups.items()):
        cfg = group[0]['config']
        if any(run['config']['num_parameters'] != cfg['num_parameters'] for run in group):
            raise ValueError('Overhead grouping requires identical model sizes; use separate results directories for modified configs.')
        row = {'n_nodes': nodes, 'num_clients': clients, 'enc_in': cfg['enc_in'],
               'params': cfg['num_parameters'], 'model_mb': cfg['model_mb'],
               'round_comm_mb_per_client': cfg['round_comm_mb_per_client'],
               'round_comm_mb_total': cfg['round_comm_mb_total']}
        # Same pooled timing quartiles as the original compute_fl_overhead.py.
        for source, name in (
            ('client_fit_train_s', 'client_train_seconds'), ('agg_s', 'agg_seconds'),
            ('server_eval_total_s', 'server_eval_seconds'), ('round_total_s', 'round_total_seconds'),
        ):
            samples = [value for run in group for value in run['timings'][source]]
            for suffix, value in zip(('q25', 'median', 'q75'), quantiles_or_nan(samples)):
                row[f'{name}_{suffix}'] = value
        row['n_log_files'] = len(group)
        rows.append(row)
    return rows


def plot_overhead(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 3.6))
    for nodes, color, marker in ((100, 'tab:orange', 'o'), (1000, 'tab:green', 's')):
        group = sorted((r for r in rows if r['n_nodes'] == nodes), key=lambda r: r['num_clients'])
        if not group:
            continue
        clients = [r['num_clients'] for r in group]
        # Match Figure 3: display model size; CSV also reports 2x size for the
        # actual per-round uplink + downlink payload.
        axes[0].plot(clients, [r['model_mb'] for r in group], marker=marker,
                     color=color, label=f'{nodes} nodes', linewidth=2)
        for ax, prefix in zip(axes[1:], ('client_train_seconds', 'agg_seconds')):
            ax.plot(clients, [r[f'{prefix}_median'] for r in group],
                    marker=marker, color=color, linewidth=2)
            ax.fill_between(clients, [r[f'{prefix}_q25'] for r in group],
                            [r[f'{prefix}_q75'] for r in group], color=color, alpha=0.22)
    for ax, title, ylabel in zip(
        axes, ('Per-Client Model Size', 'Client-Side Training Time', 'Server Aggregation Time'),
        ('Model size (MiB)', 'Time (s)', 'Time (s)'),
    ):
        ax.set(title=title, ylabel=ylabel, xlabel='Number of clients C', xscale='log')
        ax.set_xticks([1, 10, 50, 100], labels=['1', '10', '50', '100'])
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend()
    save_figure(fig, output / 'fl_overhead_quantile.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir / 'plots'
    runs = []
    for config in sorted(args.results_dir.glob('*/config.json')):
        run = read_run(config.parent)
        if run is not None and run['config']['aquacast_data_path'] in DATASETS:
            runs.append(run)
    if not runs:
        parser.error('No completed paper runs found; run federated_learning.run first.')
    output.mkdir(parents=True, exist_ok=True)
    best = {}
    for run in runs:
        cfg = run['config']
        mode = 'heterogeneous' if cfg['aquacast_partition_sizes'] is not None else 'homogeneous'
        key = (cfg['aquacast_data_path'], cfg['num_clients'], mode)
        if key not in best or run['best']['mean'] < best[key]['best']['mean']:
            best[key] = run
    expected = [(dataset, c, 'homogeneous') for dataset in DATASETS for c in (1, 10, 50, 100)]
    expected += [(dataset, 10, 'heterogeneous') for dataset in DATASETS]
    missing = [key for key in expected if key not in best]
    if missing:
        print(f'Incomplete sweep: {len(missing)} of 20 configurations missing; figures show available results.')
    summary = [
        {'dataset': dataset, 'setting': DATASETS[dataset], 'num_clients': clients,
         'partition': mode, 'best_round': run['best']['round'],
         'mse_mean': run['best']['mean'], 'mse_std': run['best']['std'], 'run': run['path']}
        for (dataset, clients, mode), run in sorted(best.items())
    ]
    write_csv(output / 'best_mse.csv', summary)
    overhead = overhead_rows(runs)
    write_csv(output / 'fl_overhead_quantile.csv', overhead)
    plt.rcParams.update({'font.size': 16})
    plot_scaling(best, output)
    plot_heterogeneity(best, output)
    plot_overhead(overhead, output)


if __name__ == '__main__':
    main()
