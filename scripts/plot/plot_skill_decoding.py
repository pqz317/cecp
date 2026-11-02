"""Partner-stage decoding accuracy over the episode, one curve per run (paper Fig 10a), from
evaluate_skill_decoding outputs. Bands = std over the cross-validation folds.

    python scripts/plot/plot_skill_decoding.py runs/oc_cecp runs/oc_cec --out figures/skill_decoding
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from cecp.eval.display import label_color
from cecp.eval.plotting import eval_dir, parse_labels, read_csv_rows, run_labels, save_fig

FONTSIZE = 18


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories evaluated with evaluate_skill_decoding.py')
    parser.add_argument('--agent-id', type=int, default=0, help='ego agent whose results to plot')
    parser.add_argument('--eval-name', default='skill_decoding')
    parser.add_argument('--labels', nargs='*', help='run_name=Label overrides')
    parser.add_argument('--out', default='figures/skill_decoding', help='output path prefix')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    chances = set()
    for run, label in zip(args.runs, run_labels(args.runs, parse_labels(args.labels))):
        agent_dir = eval_dir(run, args.eval_name) / f'agent_{args.agent_id}'
        rows = read_csv_rows(agent_dir / 'checkpoint_stage_decoding_accuracy_by_timestep.csv')
        t = np.array([int(r['timestep']) for r in rows])
        acc = np.array([float(r['accuracy_all_latent_dims']) for r in rows])
        folds = np.array([[float(v) for k, v in r.items() if k.startswith('fold_')] for r in rows])
        std = folds.std(axis=1)
        color = label_color(label)
        ax.fill_between(t, acc - std, acc + std, color=color, alpha=0.15)
        ax.plot(t, acc, color=color, linewidth=2.0, label=label)
        chances.add(json.loads((agent_dir / 'summary.json').read_text())['stage_chance_level'])

    for chance in chances:
        ax.axhline(chance, linestyle='--', color='gray', linewidth=1.0, label='Chance')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=min(0.3, min(chances) - 0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('Timestep in episode', fontsize=FONTSIZE)
    ax.set_ylabel('Decoding accuracy', fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=14)
    save_fig(fig, Path(args.out), dpi=300)


if __name__ == '__main__':
    main()
