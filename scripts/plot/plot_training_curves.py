"""Cooperator training curves: cross-play return against the evaluation (held-out) partners over
training, mean +- std over a run's cooperators (paper Fig 8: CECP vs CECP without prediction). Reads
each run's metrics.csv.

    python scripts/plot/plot_training_curves.py runs/oc_cecp runs/oc_cecp_no_pred --out figures/training_curves
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cecp.eval.display import label_color
from cecp.eval.plotting import parse_labels, run_labels, save_fig

DEFAULT_KEY = r'cooperator(-\d+)?_eval/crossplay-return'


def read_curves(run: str, key_pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (steps, values[step, learner]) for the metrics matching `key_pattern` (one per learner)."""
    pattern = re.compile(key_pattern + '$')
    by_key: dict[str, dict[int, float]] = defaultdict(dict)
    with open(Path(run) / 'metrics.csv', newline='') as f:
        for row in csv.DictReader(f):
            if pattern.match(row['key']) and row['step'] != '':
                by_key[row['key']][int(row['step'])] = float(row['value'])
    if not by_key:
        raise ValueError(f'No metrics matching {key_pattern!r} in {run}/metrics.csv')
    steps = sorted(set.intersection(*(set(v) for v in by_key.values())))
    return np.array(steps), np.array([[curve[s] for curve in by_key.values()] for s in steps])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='training run directories')
    parser.add_argument('--key', default=DEFAULT_KEY, help='regex of the per-learner metric keys to plot')
    parser.add_argument('--labels', nargs='*', help='run_name=Label overrides')
    parser.add_argument('--ylabel', default='Return on held-out envs/partners')
    parser.add_argument('--out', default='figures/training_curves', help='output path prefix')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    for run, label in zip(args.runs, run_labels(args.runs, parse_labels(args.labels))):
        steps, values = read_curves(run, args.key)
        mean, std = values.mean(axis=1), values.std(axis=1)
        color = label_color(label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.plot(steps, mean, color=color, linewidth=2.0, label=label)
    ax.set_xlabel('Training step (per agent)')
    ax.set_ylabel(args.ylabel)
    ax.legend(frameon=False)
    save_fig(fig, Path(args.out), dpi=300)


if __name__ == '__main__':
    main()
