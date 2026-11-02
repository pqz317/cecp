"""Helpers for the plotting scripts in scripts/plot/ (no JAX: they only read eval outputs)."""
import csv
from pathlib import Path
from typing import Any, Sequence

from .display import run_label

EVAL_DIR = 'eval'


def eval_dir(run_path: str | Path, eval_name: str) -> Path:
    """<run>/eval/<eval_name>/, which must already exist."""
    path = Path(run_path) / EVAL_DIR / eval_name
    if not path.is_dir():
        raise FileNotFoundError(f'{path} not found; run the matching scripts/eval/ script on {run_path} first.')
    return path


def parse_labels(items: Sequence[str] | None) -> dict[str, str]:
    """Parse `run_name=Label` command-line items."""
    labels = {}
    for item in items or []:
        name, sep, label = item.partition('=')
        if not sep:
            raise ValueError(f'--labels items must look like run_name=Label, got {item!r}')
        labels[name] = label
    return labels


def run_labels(run_paths: Sequence[str | Path], labels: dict[str, str] | None = None) -> list[str]:
    """Display label per run; runs that would share a label are told apart by their run name."""
    names = [Path(p).name for p in run_paths]
    out = [run_label(name, labels) for name in names]
    return [f'{label} ({name})' if out.count(label) > 1 else label for label, name in zip(out, names)]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def save_fig(fig: Any, path_stem: str | Path, formats: Sequence[str] = ('png', 'svg'), dpi: int = 200) -> None:
    """Save a matplotlib figure as <path_stem>.<fmt> for each format, then close it."""
    import matplotlib.pyplot as plt
    Path(path_stem).parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(f'{path_stem}.{fmt}', dpi=dpi, bbox_inches='tight')
        print(f'Wrote {path_stem}.{fmt}')
    plt.close(fig)
