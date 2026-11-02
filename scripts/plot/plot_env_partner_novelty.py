"""2x2 grid of familiar/novel partner x familiar/novel environment returns (paper Fig 4), from
evaluate_env_partner_novelty outputs. Bars = mean over agents, error bars = std over agents.

    python scripts/plot/plot_env_partner_novelty.py runs/cf_cec runs/cf_cecp --out figures/env_partner_novelty
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cecp.eval.display import label_color
from cecp.eval.plotting import eval_dir, parse_labels, read_csv_rows, run_labels, save_fig

PANELS = [('familiar', 'familiar'), ('novel', 'familiar'), ('familiar', 'novel'), ('novel', 'novel')]  # (partner, env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories evaluated with evaluate_env_partner_novelty.py')
    parser.add_argument('--eval-name', default='env_partner_novelty')
    parser.add_argument('--labels', nargs='*', help='run_name=Label overrides')
    parser.add_argument('--out', default='figures/env_partner_novelty', help='output path prefix')
    args = parser.parse_args()

    labels = run_labels(args.runs, parse_labels(args.labels))
    results = [{(r['partner_novelty'], r['env_novelty']): r for r in read_csv_rows(eval_dir(run, args.eval_name) / 'results.csv')}
               for run in args.runs]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for ax, (partner, env) in zip(axes.flat, PANELS):
        for i, (label, res) in enumerate(zip(labels, results)):
            row = res[(partner, env)]
            ax.bar(i, float(row['mean']), yerr=float(row['std']), width=0.7, color=label_color(label), capsize=4, alpha=0.85)
        ax.set_title(f'{partner.capitalize()} partner, {env} env')
        ax.set_xticks([])
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Average return')
    handles = [plt.Rectangle((0, 0), 1, 1, color=label_color(label)) for label in labels]
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, Path(args.out))


if __name__ == '__main__':
    main()
