import dataclasses
import csv

import jax
import numpy as np
import pytest

from cecp.config import PredictiveConfig
from cecp.controller.base import BaseController
from cecp.partners import build_manifest
from cecp.policy.build import build_actor_critic
from cecp.envs.make import make_env
from cecp.runs import RunDir, load_policy


def _run_config(env_cfg, arch, use_pred):
    return {
        'general': {'num_players': 2, 'game_length': 60, 'seed': 0},
        'env': dataclasses.asdict(env_cfg),
        'architecture': dataclasses.asdict(arch),
        'predictive': {'use_predictive_objective': use_pred, 'pred_loss_coef': 20.0},
    }


def test_create_open_and_params_roundtrip(tmp_path, cf_env_cfg, cf_arch):
    run = RunDir.create(tmp_path / 'cf_cecp', _run_config(cf_env_cfg, cf_arch, True))
    with pytest.raises(FileExistsError):
        RunDir.create(tmp_path / 'cf_cecp', run.config)
    reopened = RunDir.open(tmp_path / 'cf_cecp')
    assert reopened.config == run.config and reopened.name == 'cf_cecp'

    env, _, _ = make_env(cf_env_cfg, 2, 60)
    model = build_actor_critic(cf_arch, PredictiveConfig(True), env.get_observation_space()[0], env.get_action_space()[0])
    state_a = model.init_model_state(jax.random.key(1))
    state_b = model.init_model_state(jax.random.key(2))
    for step in (200, 0, 100):
        run.save_params(0, step, state_b)
    run.save_params(0, 'best', state_a)
    run.save_params(3, 'best', state_b)

    assert run.agent_ids() == [0, 3]
    ckpts = run.list_checkpoints(0)
    assert ckpts['best'] is not None and [s for s, _ in ckpts['steps']] == [0, 100, 200]
    assert run.list_checkpoints(3)['steps'] == []

    restored = run.load_params(0, 'best', model.init_model_state(jax.random.key(9)))
    for a, b in zip(jax.tree_util.tree_leaves(state_a), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    model_fn, loaded = load_policy(tmp_path / 'cf_cecp', 0, 'best')
    assert model_fn.has_predictive_head()
    for a, b in zip(jax.tree_util.tree_leaves(state_a), jax.tree_util.tree_leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_partner_manifest_roundtrip(tmp_path, cf_env_cfg, cf_arch):
    pop = RunDir.create(tmp_path / 'cf_cec', _run_config(cf_env_cfg, cf_arch, False))
    params = {'actor_critic_params': {'w': np.zeros(2, np.float32)}}
    for agent in range(5):
        pop.save_params(agent, 'best', params)
        for step in range(0, 1100, 100):
            pop.save_params(agent, step, params)
    manifest = build_manifest([pop], num_checkpoints_per_agent=3, include_best=True, heldout_ratio=0.2, heldout_seed=0)
    assert len(manifest.train) + len(manifest.heldout) == 5 * 4
    assert len({e.agent_id for e in manifest.heldout}) == 1
    assert sorted({e.ckpt for e in manifest.train}, key=str) == sorted({'best', 0, 500, 1000}, key=str)

    coop = RunDir.create(tmp_path / 'cf_cecp', _run_config(cf_env_cfg, cf_arch, True))
    coop.write_partners(manifest)
    assert coop.read_partners() == manifest


def test_controller_metrics_csv(tmp_path, cf_env_cfg, cf_arch):
    run = RunDir.create(tmp_path / 'r', _run_config(cf_env_cfg, cf_arch, False))

    class _C(BaseController):
        pass

    c = _C()
    c.init_run(run)
    c.log({'return': np.array([1.0, 3.0]), 'n': 4}, step=10, log_prefix='eval')
    c.log({'loss': jax.numpy.float32(0.5)}, step=20)
    with open(run.path / 'metrics.csv') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['step', 'key', 'value']
    assert rows[1:] == [['10', 'eval/return', '2.0'], ['10', 'eval/n', '4'], ['20', 'loss', '0.5']]
