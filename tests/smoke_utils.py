"""Helpers for the end-to-end training smoke tests (run the real scripts in subprocesses)."""
import csv
import math
import subprocess
import sys
from pathlib import Path

import jax

from cecp.runs import RunDir, load_policy

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def run_script(script: str, *overrides: str, timeout: int = 3600) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *overrides]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f'{" ".join(cmd)} failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}')


def read_metrics(run: RunDir) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    with open(run.path / 'metrics.csv') as f:
        for row in csv.DictReader(f):
            metrics.setdefault(row['key'], []).append(float(row['value']))
    return metrics


def check_run(run_path: Path, num_agents: int, min_step_ckpts: int, required_metric_keys: list[str]) -> RunDir:
    """Common assertions on a finished training run; returns the opened RunDir."""
    run = RunDir.open(run_path)
    assert run.agent_ids() == list(range(num_agents))
    for agent_id in run.agent_ids():
        ckpts = run.list_checkpoints(agent_id)
        assert ckpts['best'] is not None, f'agent {agent_id} has no best checkpoint'
        assert len(ckpts['steps']) >= min_step_ckpts, f'agent {agent_id}: {len(ckpts["steps"])} step checkpoints'
        model_fn, model_state = load_policy(run, agent_id, 'best')
        assert model_fn.has_predictive_head() == bool(run.config['predictive']['use_predictive_objective'])
        assert all(bool(jax.numpy.isfinite(x).all()) for x in jax.tree_util.tree_leaves(model_state))
    metrics = read_metrics(run)
    for key in required_metric_keys:
        assert key in metrics, f'missing metric {key}; have e.g. {sorted(metrics)[:10]}'
    assert all(math.isfinite(v) for values in metrics.values() for v in values), 'non-finite metric'
    return run
