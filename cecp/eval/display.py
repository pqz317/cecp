"""Labels and colours for figures.

Labels are keyed by the run names of configs/experiment/. Note that in this repo `oc_cecp` is the
predictive (paper) CECP model and `oc_cecp_no_pred` the ablation.
"""
DEFAULT_LABELS: dict[str, str] = {
    'cf_cec': 'CEC',
    'oc_cec': 'CEC',
    'cf_cecp': 'CECP',
    'oc_cecp': 'CECP',
    'oc_cecp_no_pred': 'CECP no pred',
}

LABEL_ORDER: list[str] = ['CEC', 'CECP no pred', 'CECP']

LABEL_COLORS: dict[str, str] = {
    'CEC': 'tab:green',
    'CECP no pred': 'tab:gray',
    'CECP': 'tab:red',
}

OC_ORIGINAL_LAYOUT_NAMES: dict[str, str] = {
    'cramped_room_9': 'Cramped Room',
    'asymm_advantages_9': 'Asymmetric Advantages',
    'coord_ring_9': 'Coordination Ring',
    'counter_circuit_9': 'Counter Circuit',
    'forced_coord_9': 'Forced Coordination',
}


def run_label(run_name: str, labels: dict[str, str] | None = None) -> str:
    """Display label for a run: `labels` first, then the defaults above, else the run name itself."""
    if labels and run_name in labels:
        return labels[run_name]
    return DEFAULT_LABELS.get(run_name, run_name)


def label_color(label: str, default: str = 'tab:blue') -> str:
    return LABEL_COLORS.get(label, default)


def sort_labels(labels: list[str]) -> list[str]:
    """Known labels in LABEL_ORDER first, then the rest in their given order."""
    known = [label for label in LABEL_ORDER if label in labels]
    return known + [label for label in labels if label not in LABEL_ORDER]
