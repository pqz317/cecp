import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cecp.config import PredictiveConfig
from cecp.envs.make import make_env
from cecp.policy.actor_critic import ActorCriticPPOTrainer, PredictiveSharedActorCriticModel, SharedActorCriticModel
from cecp.policy.build import build_actor_critic
from cecp.trainers.ppo import PPOTransition


def _spaces(env_cfg):
    env, _, _ = make_env(env_cfg, num_players=2, game_length=60)
    return env.get_observation_space()[0], env.get_action_space()[0]


@pytest.mark.parametrize('use_pred', [False, True])
@pytest.mark.parametrize('env_name', ['cf', 'oc'])
def test_forward_and_ppo_update(env_name, use_pred, cf_env_cfg, oc_env_cfg, cf_arch, oc_arch):
    env_cfg, arch = (cf_env_cfg, cf_arch) if env_name == 'cf' else (oc_env_cfg, oc_arch)
    obs_space, action_space = _spaces(env_cfg)
    model = build_actor_critic(arch, PredictiveConfig(use_predictive_objective=use_pred, pred_loss_coef=1000.0), obs_space, action_space)
    assert isinstance(model, PredictiveSharedActorCriticModel if use_pred else SharedActorCriticModel)
    assert model.has_predictive_head() == use_pred

    model_state = model.init_model_state(jax.random.key(0))
    B, L = 3, 4
    rnn_state = model.init_rnn_state(jax.random.key(1), batch_size=B)
    obs = obs_space.example(batch_shape=(B,))
    _, pi, val, info = model.forward_with_aux(jax.random.key(2), model_state, rnn_state, obs)
    assert val.shape == (B,)
    assert info['readout'].shape == (B, arch.readout_size)
    if use_pred:
        action = action_space.example(batch_shape=(B,))
        _, _, _, pred_obs = model.forward_with_prediction(jax.random.key(3), model_state, rnn_state, obs, action)
        for key in obs_space.keys():
            assert pred_obs[key].shape == obs[key].shape

    # One PPO gradient step on a dummy [B, L] batch.
    trainer = ActorCriticPPOTrainer(model, lr=3e-4, pred_loss_coef=1000.0 if use_pred else 0.0, minibatch_seq_len=L)
    trainer_state = trainer.init_trainer_state(jax.random.key(4))
    seq_obs = obs_space.example(batch_shape=(B, L))
    buffer = PPOTransition(
        obs=seq_obs,
        action=action_space.example(batch_shape=(B, L)),
        reward=jnp.zeros((B, L)),
        done=jnp.zeros((B, L), dtype=bool),
        log_p=jnp.zeros((B, L)),
        val=jnp.zeros((B, L)),
    )
    seq_rnn = jax.tree_util.tree_map(lambda x: jnp.repeat(x[:, None], L, axis=1), rnn_state)
    aux = {'agent_state': seq_rnn}
    if use_pred:
        aux['next_obs'] = seq_obs
    new_state, log = trainer.model_gradient_update(trainer_state, buffer, jnp.ones((B, L)), jnp.ones((B, L)), aux, jax.random.key(5))
    assert np.isfinite(float(log['loss'])) and np.isfinite(float(log['grad_norm']))
    assert ('pred_loss' in log) == use_pred
    before = jax.tree_util.tree_leaves(trainer_state['actor_critic_train_state'].params)
    after = jax.tree_util.tree_leaves(new_state['actor_critic_train_state'].params)
    assert any(not np.allclose(a, b) for a, b in zip(before, after))
