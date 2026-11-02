"""Choosing the frozen partner pool a CECP cooperator trains against (paper Sec. 4, Stage 2).

Each partner agent of a CEC population contributes its 'best' checkpoint and/or a few training
checkpoints (early/middle/late), and a fraction of partner *agents* (all of their checkpoints together)
is held out for evaluation.
"""
import random
from typing import Sequence, TypeVar

from .runs import PartnerEntry, PartnerManifest, RunDir

T = TypeVar('T')


def select_evenly_spaced_steps(step_entries: Sequence[tuple[int, T]], num_to_select: int) -> list[tuple[int, T]]:
    """Pick `num_to_select` entries spread evenly over `step_entries` (sorted by step), always including
    the first and last when selecting two or more."""
    if num_to_select <= 0 or len(step_entries) == 0:
        return []
    if num_to_select >= len(step_entries):
        return list(step_entries)
    if num_to_select == 1:
        return [step_entries[len(step_entries) // 2]]

    selected_indices: list[int] = []
    last_index = len(step_entries) - 1
    for i in range(num_to_select):
        selected_index = round(i * last_index / (num_to_select - 1))
        if selected_index not in selected_indices:
            selected_indices.append(selected_index)
    return [step_entries[index] for index in selected_indices]


def select_nearest_steps(step_entries: Sequence[tuple[int, T]], target_steps: Sequence[int]) -> list[tuple[int, T]]:
    """For each target step, pick the entry with the nearest step (duplicates dropped)."""
    if not step_entries or not target_steps:
        return []
    selected: list[tuple[int, T]] = []
    seen_indices: set[int] = set()
    for target in target_steps:
        best_idx = min(range(len(step_entries)), key=lambda i: abs(step_entries[i][0] - target))
        if best_idx not in seen_indices:
            seen_indices.add(best_idx)
            selected.append(step_entries[best_idx])
    return selected


def split_heldout(
    entries: Sequence[PartnerEntry],
    heldout_ratio: float,
    heldout_seed: int,
) -> tuple[list[PartnerEntry], list[PartnerEntry]]:
    """Split partner checkpoints into (train, heldout) by partner agent, so every checkpoint of a held-out
    agent is held out. round(K * ratio) of the K agents are held out, clamped to [1, K-1]."""
    if not 0.0 <= heldout_ratio < 1.0:
        raise ValueError(f'heldout_ratio must be in [0, 1), got {heldout_ratio}')

    entries = list(entries)
    if len(entries) == 0:
        raise ValueError('Cannot split an empty partner population.')
    if heldout_ratio == 0.0 or len(entries) == 1:
        return entries, []

    agent_keys = sorted({(e.run, e.agent_id) for e in entries})
    if len(agent_keys) == 1:
        return entries, []

    num_heldout = max(1, int(round(len(agent_keys) * heldout_ratio)))
    num_heldout = min(num_heldout, len(agent_keys) - 1)

    shuffled_keys = list(agent_keys)
    random.Random(heldout_seed).shuffle(shuffled_keys)
    heldout_keys = set(shuffled_keys[:num_heldout])

    train = [e for e in entries if (e.run, e.agent_id) not in heldout_keys]
    heldout = [e for e in entries if (e.run, e.agent_id) in heldout_keys]
    return train, heldout


def build_manifest(
    partner_runs: Sequence[RunDir],
    num_checkpoints_per_agent: int,
    include_best: bool = True,
    target_steps: Sequence[int] = (),
    heldout_ratio: float = 0.0,
    heldout_seed: int = 0,
) -> PartnerManifest:
    """Select partner checkpoints from one or more CEC population runs and split off held-out agents.

    Per agent: its 'best' checkpoint (if `include_best`), then the step checkpoints nearest
    `target_steps` if given, otherwise `num_checkpoints_per_agent` evenly spaced ones.
    """
    entries: list[PartnerEntry] = []
    for run in partner_runs:
        agent_ids = run.agent_ids()
        if not agent_ids:
            raise ValueError(f'No agent checkpoints found in partner run {run.path}')
        for agent_id in agent_ids:
            ckpts = run.list_checkpoints(agent_id)
            if include_best:
                if ckpts['best'] is None:
                    raise ValueError(f"include_best=True but {run.path} agent {agent_id} has no 'best' checkpoint")
                entries.append(PartnerEntry(run=str(run.path), agent_id=agent_id, ckpt='best'))
            if target_steps:
                selected = select_nearest_steps(ckpts['steps'], target_steps)
            else:
                selected = select_evenly_spaced_steps(ckpts['steps'], num_checkpoints_per_agent)
            entries.extend(PartnerEntry(run=str(run.path), agent_id=agent_id, ckpt=step) for step, _ in selected)
    if not entries:
        raise ValueError('Partner selection produced no checkpoints.')

    train, heldout = split_heldout(entries, heldout_ratio, heldout_seed)
    return PartnerManifest(
        partner_runs=[str(r.path) for r in partner_runs],
        num_checkpoints_per_agent=int(num_checkpoints_per_agent),
        include_best=bool(include_best),
        target_steps=[int(s) for s in target_steps],
        heldout_ratio=float(heldout_ratio),
        heldout_seed=int(heldout_seed),
        train=train,
        heldout=heldout,
    )
