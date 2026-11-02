"""Partner-stage I(a; z | s), true vs shuffled stage labels, per layout (paper Fig 18), from
evaluate_partner_stage_mi outputs. Bars = mean over ego agents (s.e.m. error bars) of the true value
and of the permutation-null mean.

Significance per (layout, run), true vs shuffle paired over egos, written to <out>_significance.csv:
  wilcoxon_p          one-sided Wilcoxon signed-rank on the per-ego differences (true - null mean)
  paired_t_p          one-sided paired t-test on the same differences
  max_permutation_p   the largest per-ego permutation p, (1 + #{null >= true}) / (1 + n_permutations);
                      the stars on the figure follow this one

    python scripts/plot/plot_partner_stage_mi_bar.py runs/oc_cecp --out figures/partner_stage_mi
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cecp.eval.display import OC_ORIGINAL_LAYOUT_NAMES, label_color
from cecp.eval.plotting import eval_dir, parse_labels, read_csv_rows, run_labels, save_fig

METRICS = ('weighted', 'uniform')    # column order of each shard's null.npy
YLABELS = {'weighted': 'I(a; z | s)  [nats]', 'uniform': 'I(a; z | s), uniform p(s)  [nats]'}
FONTSIZE = 18


def stars(p: float) -> str:
    if not np.isfinite(p):
        return 'n/a'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def one_sided_p(diffs: np.ndarray, test: str) -> float:
    if diffs.size < 2 or np.allclose(diffs, 0.0) or (test == 't' and np.isclose(diffs.std(ddof=1), 0.0)):
        return float('nan')
    if test == 'wilcoxon':
        return float(stats.wilcoxon(diffs, alternative='greater').pvalue)
    return float(stats.ttest_1samp(diffs, 0.0, alternative='greater').pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='ego runs evaluated with evaluate_partner_stage_mi.py, in bar order')
    parser.add_argument('--eval-name', default='partner_stage_mi')
    parser.add_argument('--metric', default='uniform', choices=METRICS)
    parser.add_argument('--labels', nargs='*', help='run_name=Label overrides')
    parser.add_argument('--out', default='figures/partner_stage_mi', help='output path prefix')
    args = parser.parse_args()
    col, m = METRICS.index(args.metric), args.metric

    labels = run_labels(args.runs, parse_labels(args.labels))
    data, sig_rows = {}, []   # (layout, label) -> (true [egos], null mean [egos])
    for run, label in zip(args.runs, labels):
        d = eval_dir(run, args.eval_name)
        for layout in sorted({r['layout'] for r in read_csv_rows(d / 'partner_stage_mi_by_ego.csv')}):
            rows = [r for r in read_csv_rows(d / 'partner_stage_mi_by_ego.csv') if r['layout'] == layout]
            true = np.array([float(r[f'mi_{m}_true']) for r in rows])
            null_mean = np.array([float(r[f'mi_{m}_null_mean']) for r in rows])
            perm_p = []
            for r, t in zip(rows, true):
                null = np.load(d / layout / f'agent_{r["ego_agent_id"]}' / 'null.npy')[:, col]
                null = null[np.isfinite(null)]
                perm_p.append((1 + np.sum(null >= t)) / (1 + null.size))
            data[(layout, label)] = (true, null_mean)
            diffs = true - null_mean
            max_p = float(np.max(perm_p))
            sig_rows.append({'layout': layout, 'label': label, 'metric': m, 'n_egos': len(rows), 'true_mean': float(true.mean()),
                             'shuffle_mean': float(null_mean.mean()), 'excess_mean': float(diffs.mean()),
                             'n_egos_true_above_shuffle': int((diffs > 0).sum()), 'wilcoxon_p': one_sided_p(diffs, 'wilcoxon'),
                             'paired_t_p': one_sided_p(diffs, 't'), 'max_permutation_p': max_p, 'stars': stars(max_p)})
            print(f'{layout:<20} {label:<8} true={true.mean():.5f} shuffle={null_mean.mean():.5f} '
                  f'max permutation p={max_p:.3g} {stars(max_p)}')

    layouts = sorted({layout for layout, _ in data})
    n_bars = 2 * len(labels)
    width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(1.5 * len(layouts) + 1.5, 4.5))
    tops: dict[tuple, list] = {}
    for i, label in enumerate(labels):
        color = label_color(label, f'C{i}')
        for kind_idx, kind in enumerate(('true', 'shuffle')):
            values = [data[(layout, label)][kind_idx] for layout in layouts]
            means = np.array([v.mean() for v in values])
            sems = np.array([v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0 for v in values])
            x = np.arange(len(layouts)) + (2 * i + kind_idx - (n_bars - 1) / 2) * width
            ax.bar(x, means, width, yerr=sems, capsize=2, color=color if kind == 'true' else 'white', edgecolor=color,
                   linewidth=1.2, hatch=None if kind == 'true' else '///', label=f'{label} {kind}')
            for layout, xi, top in zip(layouts, x, means + sems):
                tops.setdefault((layout, label), []).append((float(xi), float(top)))
    # Bracket over each true/shuffle pair, labelled with the permutation-test stars.
    y_max = max(top for pairs in tops.values() for _, top in pairs)
    gap = y_max * 0.04
    sig = {(r['layout'], r['label']): r for r in sig_rows}
    for key, ((x0, t0), (x1, t1)) in tops.items():
        y = max(t0, t1) + gap
        ax.plot([x0, x0, x1, x1], [y, y + gap * 0.5, y + gap * 0.5, y], color='black', linewidth=0.9)
        ax.text((x0 + x1) / 2, y + gap * 0.7, sig[key]['stars'], ha='center', va='bottom', fontsize=FONTSIZE)
    ax.set_ylim(top=y_max + gap * 4)
    ax.set_xticks(np.arange(len(layouts)), [OC_ORIGINAL_LAYOUT_NAMES.get(lay, lay) for lay in layouts],
                  rotation=30, ha='right', fontsize=FONTSIZE)
    ax.tick_params(axis='y', labelsize=FONTSIZE)
    ax.set_ylabel(YLABELS[m], fontsize=FONTSIZE)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    out = Path(args.out)
    save_fig(fig, out)

    import csv
    with open(f'{out}_significance.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sig_rows[0]))
        writer.writeheader()
        writer.writerows(sig_rows)
    print(f'Wrote {out}_significance.csv')


if __name__ == '__main__':
    main()
