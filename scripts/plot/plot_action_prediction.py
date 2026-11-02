"""Partner-movement prediction accuracy (paper Fig 9a, AI side), from evaluate_action_prediction outputs:
accuracy over the episode (band = std over partners) and the overall accuracy as a bar (error bar =
std of the per-partner accuracies). The comparison with human participants is made in the human-study
repository.

    python scripts/plot/plot_action_prediction.py runs/oc_cecp --out figures/action_prediction
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cecp.eval.display import label_color
from cecp.eval.plotting import eval_dir, parse_labels, read_csv_rows, run_labels, save_fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories evaluated with evaluate_action_prediction.py')
    parser.add_argument('--eval-name', default='action_prediction')
    parser.add_argument('--labels', nargs='*', help='run_name=Label overrides')
    parser.add_argument('--out', default='figures/action_prediction', help='output path prefix')
    args = parser.parse_args()
    labels = run_labels(args.runs, parse_labels(args.labels))
    dirs = [eval_dir(run, args.eval_name) for run in args.runs]
    chance = json.loads((dirs[0] / 'summary.json').read_text())['chance_level']

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for d, label in zip(dirs, labels):
        overall = np.load(d / 'accuracy_by_timestep.npy')
        std = np.load(d / 'per_partner_accuracy_by_timestep.npy').std(axis=0)
        t = np.arange(len(overall))
        color = 'black' if len(dirs) == 1 else label_color(label)
        ax.fill_between(t, overall - std, overall + std, color=color, alpha=0.15)
        ax.plot(t, overall, color=color, linewidth=2.0, label=label)
    ax.axhline(chance, linestyle='--', color='gray', linewidth=1.0, label=f'Chance ({chance:.2f})')
    ax.set(xlim=(0, len(t) - 1), xlabel='Timestep in episode', ylabel='Prediction accuracy',
           title='Partner movement prediction accuracy')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    save_fig(fig, Path(f'{args.out}_by_timestep'), dpi=300)

    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(dirs), 4), constrained_layout=True)
    for i, (d, label) in enumerate(zip(dirs, labels)):
        overall_acc = json.loads((d / 'summary.json').read_text())['overall_accuracy']
        per_partner = [float(r['overall_accuracy']) for r in read_csv_rows(d / 'partners.csv')]
        ax.bar(i, overall_acc, yerr=np.std(per_partner), color=label_color(label), capsize=4, alpha=0.85, width=0.6)
        print(f'{label}: overall accuracy {overall_acc:.4f} (std over {len(per_partner)} partners {np.std(per_partner):.4f})')
    ax.axhline(chance, linestyle='--', color='gray', linewidth=1.0)
    ax.set_xticks(range(len(dirs)), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Prediction accuracy')
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, Path(f'{args.out}_bar'), dpi=300)


if __name__ == '__main__':
    main()
