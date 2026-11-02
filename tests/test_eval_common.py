import dataclasses

import jax
import numpy as np
import pytest

from cecp.config import CoopForagingEnvConfig
from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PartnerPoolConfig, PolicyLoader, check_same_env, eval_env_const, eval_output_dir, make_eval_env, partner_pool,
    read_eval_config, resolve_partner_pool, write_matrix_csv,
)
from cecp.eval.rollouts import compile_pair_evaluator
from cecp.partners import build_manifest
from cecp.runs import RunDir, build_model_fn


def _config(env_cfg, arch=None, game_length=60):
    cfg = {'general': {'num_players': 2, 'game_length': game_length, 'seed': 0}, 'env': dataclasses.asdict(env_cfg)}
    if arch is not None:
        cfg |= {'architecture': dataclasses.asdict(arch),
                'predictive': {'use_predictive_objective': False, 'pred_loss_coef': 1.0}}
    return cfg


def _fake_run(path, env_cfg, steps_per_agent):
    """A run directory with placeholder checkpoint files (enough for checkpoint selection)."""
    run = RunDir.create(path, _config(env_cfg))
    for agent_id, steps in enumerate(steps_per_agent):
        for ckpt in ['best', *steps]:
            p = run.ckpt_dir(agent_id, ckpt) / 'params.msgpack'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'')
    return run


def test_eval_output_dir_refuses_overwrite(tmp_path, cf_env_cfg):
    run = RunDir.create(tmp_path / 'run', _config(cf_env_cfg))
    out = eval_output_dir(run, 'xp', {'seed': 3, 'runs': [str(run.path)]})
    assert out == run.path / 'eval' / 'xp'
    assert read_eval_config(out) == {'seed': 3, 'runs': [str(run.path)]}
    (out / 'stale.csv').write_text('x')
    with pytest.raises(FileExistsError):
        eval_output_dir(run, 'xp', {'seed': 4})
    out = eval_output_dir(run, 'xp', {'seed': 4}, overwrite=True)
    assert not (out / 'stale.csv').exists() and read_eval_config(out) == {'seed': 4}


def test_partner_pool_order_and_stages(tmp_path, cf_env_cfg):
    steps = [0, 100, 200, 300, 400]
    run = _fake_run(tmp_path / 'cf_cec', cf_env_cfg, [steps, steps])
    pool = partner_pool([run], num_checkpoints_per_agent=3, include_best=True)
    # Same order as train_cecp.py's selection: agents ascending, best first, then steps ascending.
    assert [(p.entry.agent_id, p.entry.ckpt, p.stage) for p in pool] == [
        (0, 'best', None), (0, 0, 0), (0, 200, 1), (0, 400, 2),
        (1, 'best', None), (1, 0, 0), (1, 200, 1), (1, 400, 2),
    ]
    assert pool[2].name == 'cf_cec_agent-0_step-200'
    assert [p.entry.ckpt for p in partner_pool([run], 3, include_best=False)] == [0, 200, 400, 0, 200, 400]


def test_partner_pool_rejects_uneven_stages(tmp_path, cf_env_cfg):
    run = _fake_run(tmp_path / 'cf_cec', cf_env_cfg, [[0, 100, 200], [0, 100]])
    with pytest.raises(ValueError, match='different numbers of step checkpoints'):
        partner_pool([run], num_checkpoints_per_agent=3, include_best=False)


def test_resolve_partner_pool_defaults_to_partners_yaml(tmp_path, cf_env_cfg):
    runs_dir = tmp_path / 'runs'
    cec = _fake_run(runs_dir / 'cf_cec', cf_env_cfg, [[0, 100, 200, 300, 400]] * 4)
    cecp = RunDir.create(runs_dir / 'cf_cecp', _config(cf_env_cfg))
    manifest = build_manifest([cec], num_checkpoints_per_agent=3, include_best=True, heldout_ratio=0.5)
    # As written by a run launched from another working directory: the stored path doesn't resolve from here.
    manifest.partner_runs = ['elsewhere/runs/cf_cec']
    cecp.write_partners(manifest)

    pool, selection = resolve_partner_pool(PartnerPoolConfig(include_best=False), ego_run=cecp)
    assert selection['runs'] == [str(cec.path)] and selection['num_checkpoints_per_agent'] == 3
    # Same step checkpoints as the cooperator's partner selection (training and held-out agents alike).
    trained_steps = {(e.agent_id, e.ckpt) for e in manifest.train + manifest.heldout if e.ckpt != 'best'}
    assert {(p.entry.agent_id, p.entry.ckpt) for p in pool} == trained_steps
    assert len(pool) == 12 and {p.stage for p in pool} == {0, 1, 2}

    pool, selection = resolve_partner_pool(PartnerPoolConfig(runs=[str(cec.path)], num_checkpoints_per_agent=2), ego_run=cecp)
    assert len(pool) == 4 * 3 and selection['num_checkpoints_per_agent'] == 2
    # A CEC population without partners.yaml is its own partner pool; the selection size must be given.
    with pytest.raises(ValueError, match='num_checkpoints_per_agent'):
        resolve_partner_pool(PartnerPoolConfig(), ego_run=cec)
    pool, selection = resolve_partner_pool(PartnerPoolConfig(num_checkpoints_per_agent=3), ego_run=cec)
    assert selection['runs'] == [str(cec.path)] and len(pool) == 4 * 4


def test_check_same_env(tmp_path, cf_env_cfg):
    a = RunDir.create(tmp_path / 'a', _config(cf_env_cfg))
    b = RunDir.create(tmp_path / 'b', _config(dataclasses.replace(cf_env_cfg, reward_shaping_steps=5)))
    c = RunDir.create(tmp_path / 'c', _config(dataclasses.replace(cf_env_cfg, heldout_goal_ratio=0.5)))
    check_same_env([a, b])  # reward shaping only matters for training
    with pytest.raises(ValueError, match='heldout_goal_ratio'):
        check_same_env([a, c])
    check_same_env([a, c], env_overrides={'heldout_goal_ratio': 0.3})
    with pytest.raises(ValueError, match='not in the env config'):
        check_same_env([a], env_overrides={'no_such_key': 1})


def test_make_eval_env_uses_run_config(tmp_path):
    env_cfg = CoopForagingEnvConfig(obstacle_density=0.0, heldout_goal_ratio=0.2, heldout_goal_seed=7, eval_on_reserved_goals=True)
    run = RunDir.create(tmp_path / 'run', _config(env_cfg))
    ev = make_eval_env(run, game_length=12)
    assert (ev.env_mode, ev.eval_on_reserved, ev.num_players, ev.game_length) == ('coop_foraging', True, 2, 12)
    assert not make_eval_env(run, env_overrides={'eval_on_reserved_goals': False}).eval_on_reserved


def test_write_matrix_csv(tmp_path):
    write_matrix_csv(tmp_path / 'm.csv', np.array([[1.0, 2.0], [3.0, 4.0]]), [5, 7])
    assert (tmp_path / 'm.csv').read_text().splitlines() == ['agent_id,5,7', '5,1.0,2.0', '7,3.0,4.0']


def test_policy_loader_and_pair_evaluator(tmp_path, cf_env_cfg, cf_arch):
    run = RunDir.create(tmp_path / 'run', _config(cf_env_cfg, cf_arch, game_length=8))
    model_fn, _ = build_model_fn(run)
    for agent_id in (0, 1):
        run.save_params(agent_id, 'best', model_fn.init_model_state(jax.random.key(agent_id)))

    loader = PolicyLoader()
    policies = loader.load_population(run)
    assert [p.agent_id for p in policies] == [0, 1] and policies[0].model_fn is policies[1].model_fn

    ev = make_eval_env(run)
    num_envs = 3
    eval_env = FrozenAssignmentEnv(ev.env, num_envs, env_agent_to_role=[[0, 1]] * num_envs)
    evaluate = compile_pair_evaluator(eval_env, ev.game_length, eval_env_const(eval_env, ev.env_mode, ev.eval_on_reserved),
                                      model_fn, model_fn, policies[0].model_state, policies[1].model_state)
    ret_i, ret_j = evaluate(jax.random.key(1), policies[0].model_state, policies[1].model_state)
    assert ret_i.shape == ret_j.shape == (num_envs,)
    assert np.isfinite(np.asarray(ret_i)).all()
    again_i, _ = evaluate(jax.random.key(1), policies[0].model_state, policies[1].model_state)
    np.testing.assert_array_equal(np.asarray(ret_i), np.asarray(again_i))
