"""Cross-play bar plots from evaluate_cross_play outputs (paper Fig 6a/15, CEC/CECP bars).

One bar per run: the mean cross-play team return over the off-diagonal agent pairs (dots = pairs,
error bar = s.e.m. over pairs) on the held-out environments. For Overcooked runs, a second plot shows
each original layout plus their average ("All 5"), with Welch t-tests of the reference run against the
others on "All 5".

    python scripts/plot/plot_cross_play.py runs/oc_cec runs/oc_cecp --out figures/cross_play
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cecp.eval.display import OC_ORIGINAL_LAYOUT_NAMES, label_color
from cecp.eval.plotting import eval_dir, parse_labels, run_labels, save_fig


def off_diagonal_pairs(matrix: np.ndarray) -> np.ndarray:
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return values[~np.isnan(values)]


def stars(p: float) -> str:
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def plot_heldout(labels: list[str], pair_means: list[np.ndarray], out: Path) -> list[dict]:
    fig, ax = plt.subplots(figsize=(4 + 0.8 * len(labels), 4))
    rows = []
    for i, (label, values) in enumerate(zip(labels, pair_means)):
        sem = float(stats.sem(values)) if len(values) > 1 else 0.0
        ax.bar(i, values.mean(), yerr=sem, color=label_color(label), alpha=0.85, capsize=4)
        ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(values)), values, color='black', alpha=0.3, s=10, zorder=3)
        rows.append({'label': label, 'mean': float(values.mean()), 'sem': sem, 'n_pairs': len(values)})
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel('Cross-play return')
    ax.set_title('Held-out environments')
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, out)
    return rows


def plot_original_layouts(labels: list[str], per_layout: list[dict[str, np.ndarray]], reference: str, out: Path) -> list[dict]:
    layouts = [name for name in per_layout[0] if name != 'original_avg']
    columns = [OC_ORIGINAL_LAYOUT_NAMES.get(name, name) for name in layouts] + ['All 5']
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(columns)), 5))
    all5 = {}
    for k, (label, matrices) in enumerate(zip(labels, per_layout)):
        values = [off_diagonal_pairs(matrices[name]) for name in layouts] + [off_diagonal_pairs(matrices['original_avg'])]
        all5[label] = values[-1]
        x = np.arange(len(columns)) + (k - (len(labels) - 1) / 2) * width
        ax.bar(x, [v.mean() for v in values], width, yerr=[v.std(ddof=1) if len(v) > 1 else 0.0 for v in values],
               color=label_color(label), alpha=0.85, capsize=3, label=label)
    rows = []
    if reference in all5:
        for label, values in all5.items():
            if label == reference:
                continue
            t, p = stats.ttest_ind(all5[reference], values, equal_var=False)
            rows.append({'column': 'All 5', 'reference': reference, 'other': label, 'reference_n': len(all5[reference]),
                         'other_n': len(values), 'reference_mean': float(all5[reference].mean()),
                         'other_mean': float(values.mean()), 't_stat': float(t), 'p_value': float(p), 'stars': stars(p)})
            print(f'All 5: {reference} vs {label}: t={t:.3f}, p={p:.3g} {stars(p)}')
    ax.set_xticks(range(len(columns)), columns, rotation=30)
    ax.set_ylabel('Cross-play return')
    ax.set_title('Original layouts')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, out)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories evaluated with evaluate_cross_play.py')
    parser.add_argument('--eval-name', default='cross_play')
    parser.add_argument('--labels', nargs='*', help='run_name=Label overrides')
    parser.add_argument('--reference', default='CECP', help='label to t-test against the others (original layouts)')
    parser.add_argument('--out', default='figures/cross_play', help='output path prefix')
    args = parser.parse_args()

    out = Path(args.out)
    labels = run_labels(args.runs, parse_labels(args.labels))
    dirs = [eval_dir(run, args.eval_name) for run in args.runs]
    pair_means = [off_diagonal_pairs(np.load(d / 'return_matrix.npy')) for d in dirs]
    write_rows(Path(f'{out}_heldout.csv'), plot_heldout(labels, pair_means, Path(f'{out}_heldout')))

    if all((d / 'return_matrix_original_avg.npy').exists() for d in dirs):
        per_layout = [{p.stem.removeprefix('return_matrix_'): np.load(p) for p in sorted(d.glob('return_matrix_*.npy'))} for d in dirs]
        reference = args.reference if args.reference in labels else labels[0]
        rows = plot_original_layouts(labels, per_layout, reference, Path(f'{out}_original_layouts'))
        write_rows(Path(f'{out}_original_layouts_significance.csv'), rows)


if __name__ == '__main__':
    main()
