from abc import abstractmethod

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from ..envs.spaces import Observation, Action
from ..utils import split_into_minibatch, split_rng_to_list
from .base import BaseTrainer


class PPOTransition(PyTreeNode):
    """
    Transition data for PPO training.

    [Param]
        - obs: Observation.
        - action: Action.
        - reward: Reward values.
        - done: Boolean array indicating episode termination.
        - log_p: Log probability of actions taken.
        - val: Value function output.
    """
    obs: Observation
    action: Action
    reward: jax.Array
    done: jax.Array
    log_p: jax.Array
    val: jax.Array

    def compute_advantage(self, last_val: jax.Array, gamma=0.99, gae_lam=0.95, unroll=16) -> jax.Array:
        """
        Computes the advantage function using Generalized Advantage Estimation (GAE).
        Based on "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (https://arxiv.org/abs/1506.02438).

        [Input]
            - self.obs: Observation with shape [seq_length, batch_size, *observation_shape].
            - self.action: Action with shape [seq_length, batch_size, *action_shape].
            - self.reward: Reward values with shape [seq_length, batch_size].
            - self.done: Boolean array indicating episode termination, shape [seq_length, batch_size].
            - last_val: Bootstrap value for the last time step with shape [batch_size,].
                
        [Param]
            - gamma: Discount factor for future rewards.
            - gae_lam: Smoothing factor for GAE.
            - unroll: Number of time steps to unroll during computation.

        [Output]
            - advantage: Computed advantage values with shape [seq_length, batch_size].
        
        [Notation]
            - L: seq_length
            - B: batch_size
        """
        def backward_iteration(gae_and_next_val: tuple[jax.Array, jax.Array], data: tuple[jax.Array, jax.Array, jax.Array]):
            gae, next_val = gae_and_next_val # [B], [B]
            reward, done, value = data # [B], [B], [B]
            delta = reward + gamma * (1 - done) * next_val - value
            gae = delta + gamma * gae_lam * (1 - done) * gae
            return (gae, value), gae # ([B], [B]), [B]
        
        _, advantage = jax.lax.scan(
            backward_iteration,
            (jnp.zeros_like(last_val), last_val),
            (self.reward, self.done, self.val),
            unroll=unroll,
            reverse=True
        )
        return advantage # [L, B]

class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        pi_loss_coef: float = 1.0,
        val_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        ppo_epochs: int = 10,
        ratio_clip: float = 0.2,
        value_clip: float = None,
        use_advantage_normalization: bool = True,
        minibatch_seq_len: int = 10,
        num_minibatches: int = None,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        PPO Algorithm Trainer

        To make this work for a specific model, inherit from this class and implement the model-specific gradient update method (e.g., model_gradient_update).
        The trainer_state (which is also initialized by a model-specific function init_trainer_state) usually contains the current model parameters along with the optimizer state.

        [Param] (for PPO Config)
            - pi_loss_coef: Coefficient scaling the policy loss term inside the PPO objective.
            - val_loss_coef: Coefficient that weights the value prediction error term, which is added to - the policy loss to compute the overall PPO loss
            - entropy_coef: Coefficient for the entropy term in policy loss to encourage exploration.
            - ratio_clip: Importance ratio (p / old_p) clipping threshold used to stabilize policy updates.
            - value_clip: Clipping threshold for value function updates to stabilize training.
            - gamma: Discount factor for future rewards.
            - gam_lam: Lambda parameter for Generalized Advantage Estimation (GAE).
            - ppo_epochs: Number of full passes (epochs) over the data for PPO updates.
            - use_advantage_normalization: Flag to normalize advantage estimates for improved training stability.
            - minibatch_seq_len: Length of each data chunk used in training (e.g., episode length or fixed segment length).
            - num_minibatches: Number of minibatches to divide the training data into.
            minibatch_num_chunks: Optional number of chunks per minibatch (if None, default behavior is applied).
        """

        self.ppo_config = {
            'pi_loss_coef': pi_loss_coef,
            'val_loss_coef': val_loss_coef,
            'entropy_coef': entropy_coef,
            'gamma': gamma,
            'gae_lam': gae_lam,
            'ppo_epochs': ppo_epochs,
            'ratio_clip': ratio_clip,
            'value_clip': value_clip,
            'use_advantage_normalization': use_advantage_normalization,
            'minibatch_seq_len': minibatch_seq_len,
            'num_minibatches': num_minibatches,
            'minibatch_num_chunks': minibatch_num_chunks
        }

    @abstractmethod
    def evaluate_action(
        self,
        model_state: PyTreeNode,
        obs: Observation,
        action: Action,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, float | jax.Array]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [batch_size, minibatch_seq_len, *observation_shape].
            - action: Action to be evaluated with shape [batch_size, minibatch_seq_len, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the log probability and the entropy.
        
        [Output]
            - log_p: Log probability of the taken action with shape [batch_size, minibatch_seq_len].
            - entropy: Entropy of the policy distribution with shape [batch_size, minibatch_seq_len].
            - value: Estimated value function output with shape [batch_size, minibatch_seq_len].
            - info: Extra logging/debugging info; values should be floats or scalar jax.Arrays.
        """
        ...

    def ppo_loss(
        self,
        model_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[float, dict[str, float]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - buffer: PPO transition buffer with attributes having shape [batch_size, minibatch_seq_len, *attribute_shape].
            - advantage: Advantage estimates with shape [batch_size, minibatch_seq_len].
            - target_val: Target value function with shape [batch_size, minibatch_seq_len].
            - aux_data: Dictionary containing any additional information needed for computing the policy loss.
        
        [Output]
            - loss: Scalar PPO loss value.
            - loss_info: Dictionary containing auxiliary loss-related statistics.

        [Notation]
            - B: batch_size
            - L: minibatch_seq_len
        """
        log_p, entropy, val, info = self.evaluate_action(model_state, buffer.obs, buffer.action, aux_data, rng)  # [B, L], [B, L], [B, L]
        
        ratio = jnp.exp(log_p - buffer.log_p)  # [B, L]
        unclipped_pi_obj = ratio * advantage  # [B, L]
        clipped_pi_obj = jnp.clip(ratio, 1 - self.ppo_config['ratio_clip'], 1 + self.ppo_config['ratio_clip']) * advantage  # [B, L]

        pi_loss = -jnp.minimum(unclipped_pi_obj, clipped_pi_obj)  # [B, L]
        if 'mask' in aux_data:
            pi_loss = (pi_loss * aux_data['mask']).mean()
            entropy = (entropy * aux_data['mask']).mean()
            info.update({'effective_mask_fraction': aux_data['mask'].mean()})
        else:
            pi_loss = pi_loss.mean()
            entropy = entropy.mean()
        info.update({
            'pi_loss': pi_loss,
            'entropy': entropy,
            'log_p': log_p.mean(),
            'ratio_clip_fraction': (jnp.abs(ratio - 1) > self.ppo_config['ratio_clip']).mean()
        })
        
        unclipped_val_loss = 0.5 * jnp.square(val - target_val)  # [B, L]
        if self.ppo_config['value_clip'] is not None:
            clipped_val = target_val + (val - target_val).clip(-self.ppo_config['value_clip'], self.ppo_config['value_clip'])  # [B, L]
            info['val_clip_fraction'] = (jnp.abs(val - target_val) > self.ppo_config['value_clip']).mean()
            clipped_val_loss = 0.5 * jnp.square(clipped_val - target_val)  # [B, L]
            val_loss = jnp.maximum(unclipped_val_loss, clipped_val_loss)
        else:
            val_loss = unclipped_val_loss
        if 'mask' in aux_data:
            val_loss = (val_loss * aux_data['mask']).mean()
        else:
            val_loss = val_loss.mean()
        info['val_loss'] = val_loss

        pi_loss_term = self.ppo_config['pi_loss_coef'] * pi_loss
        ppo_objective = pi_loss_term - self.ppo_config['entropy_coef'] * entropy + self.ppo_config['val_loss_coef'] * val_loss
        loss = ppo_objective
        info['ppo_loss'] = ppo_objective
        info['pi_loss_term'] = pi_loss_term
        info['ppo_loss_term'] = loss
        info['loss'] = loss
        return loss, info
    
    @abstractmethod
    def model_gradient_update(
        self,
        trainer_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        [Input]
            - trainer_state: Trainer state (e.g., model parameters, optimizer states, step, etc.).
            - buffer: PPO transition buffer with attributes having shape [batch_size, minibatch_seq_len, *attribute_shape].
            - target_val: Target value function with shape [batch_size, minibatch_seq_len].
            - advantage: Advantage estimates with shape [batch_size, minibatch_seq_len].
            - aux_data: Dictionary containing any additional information.
        
        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        ...

    def ppo_epoch(
        self,
        rng: jax.Array,
        trainer_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        Perform one PPO epoch and return the updated trainer state and logs. Note that the data in the buffer is expected to have shape [sequence_length, batch_size, *data_attribute_shape]. 
        """
        rng, rng_split = jax.random.split(rng)
        buffer_batch_stack, advantage_batch_stack, target_val_batch_stack, aux_data_batch_stack = split_into_minibatch(
            rng=rng_split,
            data=(buffer, advantage, target_val, aux_data),
            minibatch_seq_len=self.ppo_config['minibatch_seq_len'],
            num_minibatches=self.ppo_config['num_minibatches'],
            minibatch_num_chunks=self.ppo_config['minibatch_num_chunks']
        ) # Split samples into minibatches: [num_minibatches, minibatch_num_chunks, minibatch_seq_len, *data_attribute_shape]

        def ppo_minibatch_update(carry: tuple[PyTreeNode, jax.Array], data: tuple[PPOTransition, jax.Array, jax.Array, dict[str, PyTreeNode]]) -> tuple[tuple[PyTreeNode, jax.Array], dict[str, jax.Array]]:
            trainer_state, rng = carry
            buffer, advantage, target_val, aux_data = data
            rng, rng_update = jax.random.split(rng)
            new_trainer_state, optim_log = self.model_gradient_update(trainer_state, buffer, advantage, target_val, aux_data, rng_update)
            return (new_trainer_state, rng), optim_log
        (new_trainer_state, _), optim_log = jax.lax.scan(ppo_minibatch_update, (trainer_state, rng), (buffer_batch_stack, advantage_batch_stack, target_val_batch_stack, aux_data_batch_stack))
        optim_log = jax.tree_util.tree_map(jnp.mean, optim_log)
        return new_trainer_state, optim_log

    def learn(
        self,
        rng: jax.Array,
        trainer_state: PyTreeNode,
        buffer: PPOTransition,
        last_val: jax.Array,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        Run full PPO training (multiple epochs) and return the updated trainer state and logs.
        """
        advantage = buffer.compute_advantage(last_val, gamma=self.ppo_config['gamma'], gae_lam=self.ppo_config['gae_lam'])
        target_val = advantage + buffer.val
        if self.ppo_config['use_advantage_normalization']:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        def ppo_epoch_update(trainer_state: PyTreeNode, rng: jax.Array) -> tuple[PyTreeNode, dict[str, jax.Array]]:
            return self.ppo_epoch(rng, trainer_state, buffer, advantage, target_val, aux_data)
        rng, rng_ppo_epochs = split_rng_to_list(rng, self.ppo_config['ppo_epochs'])
        rng_ppo_epochs = jnp.stack(rng_ppo_epochs, axis=0)
        new_trainer_state, optim_log = jax.lax.scan(ppo_epoch_update, trainer_state, rng_ppo_epochs, self.ppo_config['ppo_epochs'])
        return new_trainer_state, optim_log
