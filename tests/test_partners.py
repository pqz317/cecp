import pytest

from cecp.partners import select_evenly_spaced_steps, select_nearest_steps, split_heldout
from cecp.runs import PartnerEntry


def test_evenly_spaced_matches_paper_selection():
    # 51 step checkpoints per agent (50 evals + the first chunk); 3 per agent -> first/middle/last.
    steps = [(i * 100, f'p{i}') for i in range(51)]
    assert [s for s, _ in select_evenly_spaced_steps(steps, 3)] == [0, 2500, 5000]
    assert [s for s, _ in select_evenly_spaced_steps(steps, 5)] == [0, 1200, 2500, 3800, 5000]
    assert select_evenly_spaced_steps(steps, 1) == [steps[25]]
    assert select_evenly_spaced_steps(steps[:2], 5) == steps[:2]
    assert select_evenly_spaced_steps(steps, 0) == []


def test_nearest_steps_dedupes():
    steps = [(0, 'a'), (100, 'b'), (200, 'c')]
    assert select_nearest_steps(steps, [10, 40, 190]) == [(0, 'a'), (200, 'c')]


@pytest.mark.parametrize('num_agents,ratio,expected', [(8, 0.2, 2), (10, 0.2, 2), (2, 0.2, 1), (5, 0.9, 4)])
def test_split_heldout_sizes_and_determinism(num_agents, ratio, expected):
    entries = [PartnerEntry('runs/pop', a, c) for a in range(num_agents) for c in ('best', 0, 50)]
    train, heldout = split_heldout(entries, ratio, heldout_seed=0)
    held_agents = {e.agent_id for e in heldout}
    assert len(held_agents) == expected
    assert all(e.agent_id not in held_agents for e in train)
    assert len(train) + len(heldout) == len(entries)
    assert split_heldout(entries, ratio, heldout_seed=0) == (train, heldout)


def test_split_heldout_zero_ratio():
    entries = [PartnerEntry('runs/pop', a, 'best') for a in range(4)]
    assert split_heldout(entries, 0.0, 0) == (entries, [])
