import jax
import jax.numpy as jnp
import numpy as np

from cecp.envs.batched import FrozenAssignmentEnv
from cecp.envs.make import make_env


def _zero_actions(env):
    return [space.example(batch_shape=(bs,)) for space, bs in zip(env.get_action_space(), env.get_agent_batch_size())]


def _jitted(env):
    # Envs are always jitted in training/eval; eager op-by-op execution is very slow on CPU.
    reset = jax.jit(lambda k: env.reset(k, env.default_const))
    step = jax.jit(lambda k, s, a: env.step(k, env.default_const, s, a)[:4])
    return reset, step


def _rollout(env, steps=5):
    reset, step = _jitted(env)
    state, obs_lst = reset(jax.random.key(0))
    for i in range(steps):
        state, obs_lst, reward_lst, done_lst = step(jax.random.key(i + 1), state, _zero_actions(env))
    return state, obs_lst, reward_lst, done_lst


def test_coop_foraging_reset_step(cf_env_cfg):
    env, mode, eval_on_reserved = make_env(cf_env_cfg, num_players=2, game_length=60)
    assert mode == 'coop_foraging' and eval_on_reserved
    _, obs_lst, reward_lst, done_lst = _rollout(env)
    assert len(obs_lst) == 2 and len(reward_lst) == 2 and len(done_lst) == 2
    for obs, space in zip(obs_lst, env.get_observation_space()):
        for key in space.keys():
            assert obs[key].shape[1:] == space[key].shape


def test_overcooked_reset_step_and_render(oc_env_cfg):
    env, mode, eval_on_reserved = make_env(oc_env_cfg, num_players=2, game_length=256)
    assert mode == 'overcooked' and eval_on_reserved and env.num_heldout_layouts >= 1
    state, obs_lst, _, _ = _rollout(env)
    (space,) = env.get_observation_space()[:1]
    for key in space.keys():
        assert obs_lst[0][key].shape[1:] == space[key].shape
    frame = env.create_frame_renderer()(state, 5)
    assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype != object


def test_frozen_assignment_batch(oc_env_cfg):
    env, _, _ = make_env(oc_env_cfg, num_players=2, game_length=256)
    num_envs = 4
    batched = FrozenAssignmentEnv(env, num_envs, env_agent_to_role=[[0, 1] for _ in range(num_envs)])
    reset, step = _jitted(batched)
    state, obs_lst = reset(jax.random.key(0))
    assert len(obs_lst) == 2
    state, obs_lst, reward_lst, done_lst = step(jax.random.key(1), state, _zero_actions(batched))
    assert all(np.asarray(r).shape == (num_envs,) for r in reward_lst)
