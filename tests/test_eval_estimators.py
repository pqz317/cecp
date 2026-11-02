"""Unit tests of the evaluation estimators on hand-built inputs (CPU, no environments)."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cecp.eval.diversity import entropy_gap_from_probs, pairwise_jsd_from_probs

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'eval'


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entropy_gap_and_pairwise_jsd():
    one_hot = np.eye(6)
    # Two members that always pick different actions: Delta H = JSD = log 2.
    probs = np.stack([np.broadcast_to(one_hot[0], (3, 4, 6)), np.broadcast_to(one_hot[1], (3, 4, 6))])
    np.testing.assert_allclose(entropy_gap_from_probs(probs)['delta_h'], np.log(2))
    np.testing.assert_allclose(pairwise_jsd_from_probs(probs), [[0, np.log(2)], [np.log(2), 0]])
    # Identical (stochastic) members: no disagreement, however uncertain each one is.
    same = np.broadcast_to(np.full(6, 1 / 6), (4, 3, 4, 6))
    stats = entropy_gap_from_probs(same)
    np.testing.assert_allclose(stats['delta_h'], 0.0, atol=1e-12)
    np.testing.assert_allclose(stats['h_bar'], np.log(6))
    # Eight deterministic members over six actions, most balanced split (2,2,1,1,1,1): 2.5 bits.
    picks = np.array([0, 0, 1, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(entropy_gap_from_probs(one_hot[picks][:, None, None, :])['delta_h'], 2.5 * np.log(2))
    with pytest.raises(AssertionError, match='not normalized'):
        entropy_gap_from_probs(np.full((2, 1, 1, 6), 0.5))


def test_observation_hash_matches_host_version():
    from cecp.eval.overcooked import hash_multipliers, hash_obs, hash_obs_numpy, structured_view
    obs = np.random.default_rng(0).integers(0, 5, size=(16, 50)).astype(np.uint8)
    obs[3] = obs[7]
    hashes = np.asarray(hash_obs(jnp.asarray(obs), hash_multipliers(50)))
    for i in range(len(obs)):
        np.testing.assert_array_equal(hashes[i], hash_obs_numpy(obs[i]))
    assert len(np.unique(structured_view(hashes))) == 15   # only the duplicated row collides


def test_kfold_lda_separates_classes():
    sd = _load_script('evaluate_skill_decoding')
    rng = np.random.default_rng(0)
    y = np.arange(60) % 3
    X = rng.normal(size=(4, 60, 5)) + 10.0 * np.eye(5)[y][None]         # (T, N, d), well separated
    acc = np.asarray(sd.kfold_lda_accuracy(jnp.asarray(X, jnp.float32), jnp.asarray(y), 3, 1e-4, 5))
    assert acc.shape == (4, 5) and acc.min() == 1.0
    noise = np.asarray(sd.kfold_lda_accuracy(jnp.asarray(rng.normal(size=(4, 60, 5)), jnp.float32), jnp.asarray(y), 3, 1e-4, 5))
    assert noise.mean() < 0.7


def test_partner_stage_mi_and_null():
    mi = _load_script('evaluate_partner_stage_mi')
    rng = np.random.default_rng(0)
    n_episodes, n_states, t_per_episode = 60, 4, 8
    episode_id = np.repeat(np.arange(n_episodes), t_per_episode).astype(np.int32)
    stage = (episode_id % 3).astype(np.int8)
    state_id = rng.integers(0, n_states, size=episode_id.size).astype(np.int32)

    def rows_with(probs):
        return mi.ShardRows(probs.astype(np.float16), episode_id, state_id, np.zeros_like(stage), stage,
                            np.zeros(episode_id.size, np.int16), np.zeros((n_states, 3), np.uint32), *[np.zeros(n_states)] * 5,
                            n_episodes, 0.0)

    cfg = mi.PartnerStageMIConfig(min_samples_per_cell=2, n_permutations=50, permutation_chunk=10)
    # The ego's action is a function of the partner stage: I = log 3 at every state, far above the null.
    rows = rows_with(np.eye(6)[stage])
    result = mi.mi_from_aggregates(*mi.aggregate(rows), cfg.min_samples_per_cell)
    np.testing.assert_allclose([result['weighted'], result['uniform']], np.log(3), rtol=1e-6)
    null = mi.null_distribution(rows, cfg, np.array([result['weighted'], result['uniform']]), seed=0)
    assert null.shape == (50, 2) and null.max() < 0.5 * np.log(3)
    # The ego ignores the stage: I = 0 exactly.
    rows = rows_with(np.broadcast_to(np.full(6, 1 / 6), (episode_id.size, 6)))
    result = mi.mi_from_aggregates(*mi.aggregate(rows), cfg.min_samples_per_cell)
    assert abs(result['uniform']) < 1e-9
