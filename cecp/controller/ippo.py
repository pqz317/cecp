from typing import Sequence

import jax
from flax.struct import PyTreeNode

from ..envs import Observation, Action, Env, EnvState, EnvConst
from ..policy.agent_wrapper import RecurrentPPOAgent
from ..trainers.ppo import PPOTrainer, PPOTransition
from ..utils import split_rng_to_list
from .base import BaseController


class IPPOController(BaseController):
    @staticmethod
    def ppo_rollout_step(
        env_fn: Env,
        ppo_agent_lst: Sequence[RecurrentPPOAgent],
        rng: jax.Array,
        model_state_lst: Sequence[PyTreeNode],
        agent_state_lst: Sequence[PyTreeNode],
        env_const: EnvConst,
        env_state: EnvState,
        obs_lst: Sequence[Observation]
    ) -> tuple[
        jax.Array,
        list[PyTreeNode],
        list[EnvState],
        list[Observation],
        list[dict[str, jax.Array | Action | Observation | PyTreeNode]],
        dict
    ]:
        """
        Interaction with the environment for one step using PPO agents to collect PPO-related data.

        [Input]
            - env_fn: Environment function that handles state transitions.
            - ppo_agent_lst: Sequence of PPO agents interacting with the environment.
            - rng: Random number generator state.
            - model_state_lst: Sequence of model states for each agent.
            - agent_state_lst: Sequence of internal agent states.
            - env_const: Environment constants/configurations.
            - env_state: Current state of the environment.
            - obs_lst: Sequence of observations for each agent.

        [Output]
            - rng: Random number generator state after one iteration.
            - next_agent_state_lst: The updated agent states after the step.
            - next_env_state: The updated environment state after the step.
            - next_obs_lst: New observations received by agents after the step.
            - data_lst: List of dictionaries containing observations, actions, log probabilities,
                values, agent states, and the next observation target (`next_obs`).
            - info: Additional environment-specific information.
        """
        next_agent_state_lst, action_lst, data_lst = [], [], []
        rng, rng_act_lst = split_rng_to_list(rng, len(ppo_agent_lst))
        for rng_act, agent_fn, model_state, agent_state, obs in zip(rng_act_lst, ppo_agent_lst, model_state_lst, agent_state_lst, obs_lst):
            next_agent_state, action, extra_data = agent_fn.step(
                rng=rng_act,
                model_state=model_state,
                agent_state=agent_state,
                obs=obs
            )
            
            next_agent_state_lst.append(next_agent_state)
            action_lst.append(action)
            data_lst.append({
                'obs': obs,
                'action': action,
                'agent_state': agent_state
            } | extra_data)  # Expected from extra_data of PPOAgent: log_p, val (+ optional predictive outputs).
        
        rng, rng_env_step = jax.random.split(rng)
        next_env_state, next_obs_lst, reward_lst, done_lst, info = env_fn.step(rng_env_step, env_const, env_state, action_lst)
        rng, rng_reset_lst = split_rng_to_list(rng, len(ppo_agent_lst))
        next_agent_state_lst = [
            agent_fn.reset_agent_state(rng_reset, next_agent_state, done)
            for rng_reset, agent_fn, next_agent_state, done in zip(
                rng_reset_lst, ppo_agent_lst, next_agent_state_lst, done_lst
            )
        ]
        for reward, done, next_obs, data in zip(reward_lst, done_lst, next_obs_lst, data_lst):
            data['reward'] = reward
            data['done'] = done
            data['next_obs'] = next_obs
        return rng, next_agent_state_lst, next_env_state, next_obs_lst, data_lst, info
    
    @staticmethod
    def ppo_update(
        ppo_trainer_lst: Sequence[PPOTrainer],
        rng: jax.Array,
        trainer_state_lst: Sequence[PyTreeNode],
        buffer_lst: Sequence[PPOTransition],
        last_val_lst: Sequence[jax.Array],
        aux_data_lst: Sequence[dict[str, PyTreeNode]]
    ) -> tuple[list[PyTreeNode], list[dict[str, jax.Array]]]:
        """
        Update the policy (contained in the trainer_state) with PPO using the current rollout buffer.

        [Input]
            - ppo_trainer_lst: Sequence of PPO trainer instances.
            - rng: Random number generator state.
            - trainer_state_lst: Sequence of trainer states before the update.
            - buffer_lst: Sequence of PPO transition buffers containing past experience (obs, action, reward, done, log_p, val).
            - last_val_lst: Sequence of estimated values at the last time step.
            - aux_data_lst: Sequence of auxiliary data dictionaries containing additional training information (like rnn_states, masks, etc).

        [Output]
            - new_trainer_state_lst: Updated trainer states after learning.
            - info_lst: List of dictionaries containing training statistics and updates.
        """
        new_trainer_state_lst, info_lst = [], []
        rng, rng_ppo_learn_lst = split_rng_to_list(rng, len(ppo_trainer_lst))
        for rng_ppo_learn, trainer_fn, trainer_state, buffer, last_val, aux_data in zip(rng_ppo_learn_lst, ppo_trainer_lst, trainer_state_lst, buffer_lst, last_val_lst, aux_data_lst):
            new_trainer_state, info = trainer_fn.learn(rng_ppo_learn, trainer_state, buffer, last_val, aux_data)
            new_trainer_state_lst.append(new_trainer_state)
            info_lst.append(info)
        return new_trainer_state_lst, info_lst
