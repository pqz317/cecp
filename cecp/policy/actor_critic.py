from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import flax.linen as nn
import flax.struct as struct
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from ..envs import Observation, Action, ObservationSpace, ActionSpace
from ..envs.spaces import ContinuousSpace
from ..models.action_distribution import ActionDistribution, get_distribution_info, build_action_distribution_layer
from ..models.feature_extractors import RecurrentFeatureExtractor
from ..models.nn_blocks import Activation, MLP
from ..models.nn_configs import CNNConfig, MLPConfig, NNConfig
from ..models.next_state_prediction import ObservationPredictor, ActionFeatureEncoder
from ..trainers.ppo import PPOTrainer, PPOTransition
from ..utils import global_norm
from .agent_wrapper import RecurrentAgent, RecurrentPPOAgent


FeatureExtractorConfig = TypeVar('FeatureExtractorConfig', bound='BaseFeatureExtractorConfig')

class PolicyLayerConfig(PyTreeNode):
    action_space: ActionSpace
    output_kernel_init_scale: float | None = None
    def build(self) -> dict[str, nn.Module]:
        return build_action_distribution_layer(
            self.action_space,
            output_kernel_init_scale=self.output_kernel_init_scale,
        )

class BaseFeatureExtractorConfig(PyTreeNode):
    # A pure configuration object for building feature extractor modules.
    obs_space: ObservationSpace
    def build(self) -> RecurrentFeatureExtractor:
        raise NotImplementedError


class ConfigurableFeatureExtractorConfig(BaseFeatureExtractorConfig):
    cnn_feature_lst: tuple[int, ...] = struct.field(default_factory=lambda: (64, 64, 64))
    cnn_kernel_lst: tuple[tuple[int, int], ...] = struct.field(default_factory=lambda: ((3, 3), (3, 3), (3, 3)))
    cnn_stride_lst: tuple[int, ...] = struct.field(default_factory=lambda: (1, 1, 1))
    encoder_fc_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=lambda: (64,))
    activation_type: Activation.Type = 'relu'
    encoder_kernel_init_scale: float | None = None
    use_rnn: bool = False
    rnn_hidden_layers: int = 1
    rnn_hidden_size: int = 64
    rnn_cell_type: Literal['gru', 'lstm'] = 'gru'
    use_rnn_layer_norm: bool = False

    def build(self) -> RecurrentFeatureExtractor:
        if len(self.cnn_feature_lst) != len(self.cnn_kernel_lst) or len(self.cnn_feature_lst) != len(self.cnn_stride_lst):
            raise ValueError(
                "cnn_feature_lst, cnn_kernel_lst, and cnn_stride_lst must have the same length "
                f"(got {len(self.cnn_feature_lst)}, {len(self.cnn_kernel_lst)}, {len(self.cnn_stride_lst)})"
            )

        embedding_extractor_config: dict[str, NNConfig] = {}
        for key, space in self.obs_space.items():
            if not isinstance(space, ContinuousSpace):
                raise NotImplementedError(
                    f"Expected continuous observation space for key '{key}', but got {type(space)}"
                )
            if len(space.shape) == 1:
                embedding_extractor_config[key] = MLPConfig(
                    hidden_size_lst=list(self.encoder_fc_hidden_size_lst),
                    activation_type=self.activation_type,
                    kernel_init_scale=self.encoder_kernel_init_scale,
                )
            elif len(space.shape) == 3:
                embedding_extractor_config[key] = CNNConfig(
                    feature_lst=self.cnn_feature_lst,
                    kernel_lst=self.cnn_kernel_lst,
                    stride_lst=self.cnn_stride_lst,
                    activation_type=self.activation_type,
                    post_mlp_hidden_size_lst=self.encoder_fc_hidden_size_lst,
                    post_mlp_activation_type=self.activation_type,
                    kernel_init_scale=self.encoder_kernel_init_scale,
                    post_mlp_kernel_init_scale=self.encoder_kernel_init_scale,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported observation shape {space.shape} for key '{key}'. "
                    "Only 1D and 3D continuous observations are supported."
                )

        return RecurrentFeatureExtractor(
            embedding_extractor_config=embedding_extractor_config,
            use_rnn=self.use_rnn,
            rnn_hidden_layers=self.rnn_hidden_layers,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_cell_type=self.rnn_cell_type,
            use_rnn_layer_norm=self.use_rnn_layer_norm,
            aggr_mlp_layers=0,
        )

class SharedActorCriticNet(nn.Module, Generic[FeatureExtractorConfig]):
    """CNN -> LSTM -> readout -> actor/critic heads, with an optional next-state prediction head.

    The predictive head decodes the next observation from the readout and the agent's own action
    (paper Fig. 1b); it shares the encoder with the policy but has no effect on acting.
    """
    feature_extractor_config: FeatureExtractorConfig
    policy_config: PolicyLayerConfig
    use_predictive_head: bool = False
    use_readout_layer: bool = False
    readout_size: int = 256
    obs_space: ObservationSpace | None = None
    action_space: ActionSpace | None = None
    actor_head_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=tuple)
    critic_head_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=tuple)
    head_activation_type: Activation.Type = 'relu'
    head_kernel_init_scale: float | None = None
    critic_output_kernel_init_scale: float | None = None

    def setup(self) -> None:
        self.feature_extractor = self.feature_extractor_config.build()
        if self.use_readout_layer:
            self.readout_layer = nn.Dense(self.readout_size)
        self.policy_layers = self.policy_config.build()
        if len(self.actor_head_hidden_size_lst) > 0:
            self.actor_head = MLP(
                hidden_size_lst=self.actor_head_hidden_size_lst,
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )
        if len(self.critic_head_hidden_size_lst) > 0:
            self.critic_head = MLP(
                hidden_size_lst=self.critic_head_hidden_size_lst,
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )
        value_layer_kernel_init = (
            nn.initializers.orthogonal(self.critic_output_kernel_init_scale)
            if self.critic_output_kernel_init_scale is not None
            else nn.initializers.lecun_normal()
        )
        self.value_layer = nn.Dense(1, kernel_init=value_layer_kernel_init)
        if self.use_predictive_head:
            if self.obs_space is None or self.action_space is None:
                raise ValueError("obs_space and action_space must be provided when use_predictive_head=True")
            self.observation_predictor = ObservationPredictor(
                obs_space=self.obs_space,
                embedding_extractor_config=self.feature_extractor.embedding_extractor_config
            )
            self.action_feature_encoder = ActionFeatureEncoder(action_space=self.action_space)

    def _apply_heads(
        self,
        policy_feature: jax.Array,
    ) -> tuple[ActionDistribution, jax.Array]:
        actor_feature = self.actor_head(policy_feature) if hasattr(self, 'actor_head') else policy_feature
        critic_feature = self.critic_head(policy_feature) if hasattr(self, 'critic_head') else policy_feature
        pi = ActionDistribution({
            action_name: policy_layer(actor_feature)
            for action_name, policy_layer in self.policy_layers.items()
        })
        val = self.value_layer(critic_feature).squeeze(-1)
        return pi, val

    def __call__(
        self,
        rnn_states: jax.Array,
        obs: Observation,
        action: Action | None = None,
        return_prediction: bool = False,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]] | tuple[PyTreeNode, ActionDistribution, jax.Array, Observation, dict[str, jax.Array]]:
        rnn_states, feature = self.feature_extractor(rnn_states, obs)
        info = {'feature': feature}
        if self.use_readout_layer:
            readout = self.readout_layer(feature)
            info['readout'] = readout
        else:
            readout = feature
        pi, val = self._apply_heads(readout)
        if self.use_predictive_head and return_prediction:
            action = pi.mode() if action is None else action
            action_feature = self.action_feature_encoder(action)
            predicted_obs = self.observation_predictor(readout, action_feature)
            return rnn_states, pi, val, predicted_obs, info
        return rnn_states, pi, val, info

    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        feature_extractor = self.feature_extractor_config.build()
        return feature_extractor.init_rnn_state(rng, batch_size)


class BaseActorCriticModel(ABC):
    @abstractmethod
    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        ...

    @abstractmethod
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        ...

    def init_actor_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        return self.init_rnn_state(rng, batch_size)

    def has_predictive_head(self) -> bool:
        return False

    @abstractmethod
    def forward(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        ...

    def forward_with_aux(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]]:
        next_rnn_state, pi, val = self.forward(rng, model_state, rnn_state, obs)
        return next_rnn_state, pi, val, {}

    @abstractmethod
    def get_action_distribution(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        ...

    @abstractmethod
    def get_value(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        ...

    @classmethod
    def output_nn(self, fn: SharedActorCriticNet, obs_space: ObservationSpace, depth: int | None = -1) -> None:
        if depth != -1:
            rnn_state = fn.init_rnn_state(jax.random.key(0), batch_size=1)
            obs_example = obs_space.example(batch_shape=(1,))
            if getattr(fn, 'use_predictive_head', False):
                action_space = getattr(fn, 'action_space', None)
                if action_space is None:
                    raise ValueError("action_space must be set when use_predictive_head=True")
                action_example = action_space.example(batch_shape=(1,))
                print(
                    fn.tabulate(
                        jax.random.key(0),
                        rnn_state,
                        obs_example,
                        action=action_example,
                        return_prediction=True,
                        depth=depth,
                    )
                )
            else:
                print(fn.tabulate(jax.random.key(0), rnn_state, obs_example, depth=depth))


class UnifiedSharedActorCriticModel(BaseActorCriticModel):
    @classmethod
    def build_configurable_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        cnn_feature_lst: tuple[int, ...] = (64, 64, 64),
        cnn_kernel_lst: tuple[tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3)),
        cnn_stride_lst: tuple[int, ...] = (1, 1, 1),
        encoder_fc_hidden_size_lst: tuple[int, ...] = (64,),
        use_rnn: bool = True,
        rnn_hidden_layers: int = 1,
        rnn_hidden_size: int = 64,
        rnn_cell_type: Literal['gru', 'lstm'] = 'gru',
        use_rnn_layer_norm: bool = False,
        activation_type: Activation.Type = 'relu',
        actor_head_hidden_size_lst: tuple[int, ...] = (),
        critic_head_hidden_size_lst: tuple[int, ...] = (),
        encoder_kernel_init_scale: float | None = None,
        head_kernel_init_scale: float | None = None,
        actor_output_kernel_init_scale: float | None = None,
        critic_output_kernel_init_scale: float | None = None,
        use_predictive_head: bool = False,
        readout_size: int = 256,
        output_nn_depth: int | None = -1,
    ) -> SharedActorCriticNet[ConfigurableFeatureExtractorConfig]:
        actor_critic_fn = SharedActorCriticNet(
            feature_extractor_config=ConfigurableFeatureExtractorConfig(
                obs_space=obs_space,
                cnn_feature_lst=cnn_feature_lst,
                cnn_kernel_lst=cnn_kernel_lst,
                cnn_stride_lst=cnn_stride_lst,
                encoder_fc_hidden_size_lst=encoder_fc_hidden_size_lst,
                activation_type=activation_type,
                encoder_kernel_init_scale=encoder_kernel_init_scale,
                use_rnn=use_rnn,
                rnn_hidden_layers=rnn_hidden_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_cell_type=rnn_cell_type,
                use_rnn_layer_norm=use_rnn_layer_norm,
            ),
            policy_config=PolicyLayerConfig(
                action_space=action_space,
                output_kernel_init_scale=actor_output_kernel_init_scale,
            ),
            use_predictive_head=use_predictive_head,
            use_readout_layer=True,
            readout_size=readout_size,
            obs_space=obs_space if use_predictive_head else None,
            action_space=action_space if use_predictive_head else None,
            actor_head_hidden_size_lst=actor_head_hidden_size_lst,
            critic_head_hidden_size_lst=critic_head_hidden_size_lst,
            head_activation_type=activation_type,
            head_kernel_init_scale=head_kernel_init_scale,
            critic_output_kernel_init_scale=critic_output_kernel_init_scale,
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn

    def __init__(self, actor_critic_fn: SharedActorCriticNet[FeatureExtractorConfig]) -> None:
        self.actor_critic_fn = actor_critic_fn

    def has_predictive_head(self) -> bool:
        return self.actor_critic_fn.use_predictive_head

    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        rnn_state = self.actor_critic_fn.init_rnn_state(rng, batch_size)
        return {'rnn_state': rnn_state}

    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        rng, rng_rnn = jax.random.split(rng)
        rnn_state_example = self.init_rnn_state(rng_rnn, batch_size=1)
        obs_example = self.actor_critic_fn.feature_extractor_config.obs_space.example(batch_shape=(1,))
        action_example = self.actor_critic_fn.policy_config.action_space.example(batch_shape=(1,))

        rng, rng_params = jax.random.split(rng)
        params = self.actor_critic_fn.init(
            rng_params,
            rnn_state_example['rnn_state'],
            obs_example,
            action=action_example,
            return_prediction=self.has_predictive_head(),
        )
        return {'actor_critic_params': params}

    def forward(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        next_rnn_state, pi, val, _ = self.actor_critic_fn.apply(
            model_state['actor_critic_params'],
            rnn_state['rnn_state'],
            obs,
            return_prediction=False,
        )
        return {'rnn_state': next_rnn_state}, pi, val

    def forward_with_aux(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]]:
        next_rnn_state, pi, val, info = self.actor_critic_fn.apply(
            model_state['actor_critic_params'],
            rnn_state['rnn_state'],
            obs,
            return_prediction=False,
        )
        return {'rnn_state': next_rnn_state}, pi, val, info

    def forward_with_prediction(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        action: Action | None = None
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, Observation]:
        next_rnn_state, pi, val, pred_obs, _ = self.forward_with_prediction_and_aux(rng, model_state, rnn_state, obs, action)
        return next_rnn_state, pi, val, pred_obs

    def forward_with_prediction_and_aux(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        action: Action | None = None
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, Observation, dict[str, jax.Array]]:
        if not self.has_predictive_head():
            raise ValueError("forward_with_prediction requires a shared actor-critic model with predictive head enabled")
        next_rnn_state, pi, val, pred_obs, info = self.actor_critic_fn.apply(
            model_state['actor_critic_params'],
            rnn_state['rnn_state'],
            obs,
            action,
            return_prediction=True
        )
        return {'rnn_state': next_rnn_state}, pi, val, pred_obs, info

    def get_action_distribution(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
    ) -> tuple[PyTreeNode, ActionDistribution]:
        next_rnn_state, pi, _ = self.forward(rng, model_state, rnn_state, obs)
        return next_rnn_state, pi

    def get_value(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
    ) -> tuple[PyTreeNode, jax.Array]:
        next_rnn_state, _, val = self.forward(rng, model_state, rnn_state, obs)
        return next_rnn_state, val


class SharedActorCriticModel(UnifiedSharedActorCriticModel):
    """CEC model: shared-encoder actor-critic without the next-state prediction head."""
    @classmethod
    def build_configurable_shared_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, **kwargs) -> SharedActorCriticNet[ConfigurableFeatureExtractorConfig]:
        return super().build_configurable_shared_actor_critic(
            obs_space=obs_space,
            action_space=action_space,
            use_predictive_head=False,
            **kwargs,
        )


class PredictiveSharedActorCriticModel(UnifiedSharedActorCriticModel):
    """CECP model: shared-encoder actor-critic with the next-state prediction head."""
    @classmethod
    def build_configurable_predictive_shared_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, **kwargs) -> SharedActorCriticNet[ConfigurableFeatureExtractorConfig]:
        return super().build_configurable_shared_actor_critic(
            obs_space=obs_space,
            action_space=action_space,
            use_predictive_head=True,
            **kwargs,
        )

class ActorWrapperAgent(RecurrentAgent):
    def __init__(self, actor_critic_model: BaseActorCriticModel) -> None:
        self.actor_critic_model = actor_critic_model

    def init_agent_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        return self.actor_critic_model.init_actor_rnn_state(rng, batch_size)

    def reset_agent_state(self, rng: jax.Array, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        # It is expected done has shape [batch_size,] and agent_state has shape [batch_size, *agent_state_shape]
        default_state = self.init_agent_state(rng, batch_size=done.shape[0])
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y, agent_state, default_state
        )

    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, Action, dict]:
        """
        Perform a single step in the environment with recurrent state updates.

        [Input]
            - rng (jax.Array): Random number generator state for stochastic policies.
            - model_state (PyTreeNode): The agent's current model state.
            - agent_state (PyTreeNode): The agent's recurrent state (e.g., RNN hidden states).
            - obs (Observation): The observation received from the environment.

        [Output]
            - next_agent_state (PyTreeNode): Updated agent state (e.g., next RNN hidden state).
            - action (Action): The action selected by the agent.
        """
        rng, rng_pi = jax.random.split(rng)
        next_actor_rnn_state, pi = self.actor_critic_model.get_action_distribution(rng_pi, model_state, agent_state, obs)
        rng, rng_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_sample)
        return next_actor_rnn_state, action, {}

class ActorCriticPPOAgent(RecurrentPPOAgent):
    def __init__(self, actor_critic_model: UnifiedSharedActorCriticModel) -> None:
        self.actor_critic_model = actor_critic_model

    def init_agent_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        return self.actor_critic_model.init_rnn_state(rng, batch_size)

    def reset_agent_state(self, rng: jax.Array, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        # It is expected done has shape [batch_size,] and agent_state has shape [batch_size, *agent_state_shape]
        default_state = self.init_agent_state(rng, batch_size=done.shape[0])
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y, agent_state, default_state
        )

    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, Action, dict[str, jax.Array | Observation]]:
        """
        Perform a single step in the environment with recurrent state updates and additional outputs for policy optimization.

        [Input]
            - rng: Random number generator state for stochastic policies.
            - model_state: The agent's current model state.
            - agent_state: The agent's recurrent state (e.g., RNN hidden states).
            - obs: The observation received from the environment.

        The batch shape of agent_state and obs should be the same.

        [Output]
            - next_agent_state: Updated agent state (e.g., next RNN hidden state).
            - action: The action selected by the agent.
            - extra_data: Dict with keys:
                - log_p: Log probability of the selected action.
                - val: Estimated value function output at the current observation.
                - pred_next_obs: Predicted next observation (only for predictive actor-critic model).
        """
        rng, rng_forward = jax.random.split(rng)
        next_agent_state, pi, val = self.actor_critic_model.forward(rng_forward, model_state, agent_state, obs)
        rng, rng_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_sample)
        log_p = pi.log_prob(action)

        if self.actor_critic_model.has_predictive_head():
            rng, rng_pred = jax.random.split(rng)
            _, _, _, pred_next_obs = self.actor_critic_model.forward_with_prediction(
                rng_pred,
                model_state,
                agent_state,
                obs,
                action
            )
            return next_agent_state, action, {'log_p': log_p, 'val': val, 'pred_next_obs': pred_next_obs}

        return next_agent_state, action, {'log_p': log_p, 'val': val}

class ActorCriticPPOTrainer(PPOTrainer):
    def __init__(
        self,
        actor_critic_model: UnifiedSharedActorCriticModel,
        lr: float = None,
        lr_schedule_steps: int = None,
        lr_schedule: str = 'linear',
        lr_warmup_ratio: float = 0.0,
        grad_clip_norm: float = 10.0,
        val_loss_coef: float = 1.0,
        pi_loss_coef: float = 1.0,
        pred_loss_coef: float = 0.0,
        entropy_coef: float = 0.0,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        ppo_epochs: int = 10,
        ratio_clip: float = 0.2,
        value_clip: float = None,
        use_advantage_normalization: bool = True,
        minibatch_seq_len: int = 10,
        num_minibatches: int = 1,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        See super().__init__() for more description.

        [Param (additional)]
            grad_clip_norm: Maximum norm for gradient clipping to prevent exploding gradients and improve stability.
            lr_schedule_steps: Total number of optimizer steps for the LR schedule. If None, a constant LR is used.
            lr_schedule: LR schedule type. 'constant' holds lr fixed; 'linear' decays linearly to 0;
                'warmup_cosine' linearly warms up to peak LR then cosine-decays to 0. Ignored when
                lr_schedule_steps is None.
            lr_warmup_ratio: Fraction of lr_schedule_steps used for linear warmup (only used when
                lr_schedule='warmup_cosine'). E.g. 0.05 = 5% warmup.
            pred_loss_coef: Weight of the next-state prediction loss (lambda in the paper's Eq. 7).
        """

        super().__init__(
            pi_loss_coef=pi_loss_coef,
            val_loss_coef=val_loss_coef,
            entropy_coef=entropy_coef,
            gamma=gamma,
            gae_lam=gae_lam,
            ppo_epochs=ppo_epochs,
            ratio_clip=ratio_clip,
            value_clip=value_clip,
            use_advantage_normalization=use_advantage_normalization,
            minibatch_seq_len=minibatch_seq_len,
            num_minibatches=num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks
        )
        self.actor_critic_model = actor_critic_model

        if lr_schedule not in ('constant', 'linear', 'warmup_cosine'):
            raise ValueError(f"lr_schedule must be 'constant', 'linear', or 'warmup_cosine', got '{lr_schedule}'")

        self.ppo_config.update({
            'lr': lr,
            'grad_clip_norm': grad_clip_norm,
            'lr_schedule_steps': lr_schedule_steps,
            'lr_schedule': lr_schedule,
            'lr_warmup_ratio': lr_warmup_ratio,
            'pred_loss_coef': pred_loss_coef,
        })
        self.actor_critic_tx = optax.chain(
            optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
            optax.adam(
                learning_rate=self._make_lr_schedule(
                    self.ppo_config['lr'],
                    self.ppo_config['lr_schedule_steps'],
                    self.ppo_config['lr_schedule'],
                    self.ppo_config['lr_warmup_ratio'],
                )
            )
        )

    @staticmethod
    def _make_lr_schedule(
        peak_lr: float,
        total_steps: int | None,
        schedule: str,
        warmup_ratio: float,
    ):
        if total_steps is None or schedule == 'constant':
            return peak_lr
        if schedule == 'warmup_cosine':
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=peak_lr,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_steps=total_steps,
                end_value=0.0,
            )
        return optax.linear_schedule(
            init_value=peak_lr,
            end_value=0.0,
            transition_steps=total_steps,
        )

    def init_trainer_state(self, rng: jax.Array) -> PyTreeNode:
        """
        Initialize the trainer state.
        """
        # NB: the model is initialized from `rng`, not the split-off key; kept as in the paper runs.
        rng, rng_init_model = jax.random.split(rng)
        model_state = self.actor_critic_model.init_model_state(rng)
        actor_critic_train_state = TrainState.create(
            apply_fn=self.actor_critic_model.actor_critic_fn.apply,
            params=model_state['actor_critic_params'],
            tx=self.actor_critic_tx
        )
        return {'actor_critic_train_state': actor_critic_train_state}

    def model_state_from_trainer_state(self, trainer_state: PyTreeNode) -> PyTreeNode:
        """
        Retreive the model state from the trainer state.
        """
        return {'actor_critic_params': trainer_state['actor_critic_train_state'].params}

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
                - agent_state: The agent's recurrent state (e.g., RNN hidden states with shape [batch_size, minibatch_seq_len, *rnn_state_shape]).
                - next_obs: (predictive model only) target next observation for the prediction loss.

        [Output]
            - log_p: Log probability of the taken action with shape [batch_size, minibatch_seq_len].
            - entropy: Entropy of the policy distribution with shape [batch_size, minibatch_seq_len].
            - value: Estimated value function output with shape [batch_size, minibatch_seq_len].
            - info: Extra logging/debugging info; values should be floats or scalar jax.Arrays.
        """
        rnn_state = {'rnn_state': jax.tree_util.tree_map(lambda x: x[:, 0], aux_data['agent_state']['rnn_state'])}
        if self.actor_critic_model.has_predictive_head():
            next_rnn_state, pi, val, pred_next_obs, model_info = self.actor_critic_model.forward_with_prediction_and_aux(
                rng,
                model_state,
                rnn_state,
                obs,
                action
            )
        else:
            next_rnn_state, pi, val, model_info = self.actor_critic_model.forward_with_aux(
                rng,
                model_state,
                rnn_state,
                obs,
            )
            pred_next_obs = None
        log_p = pi.log_prob(action)
        entropy = pi.entropy()
        info = get_distribution_info(pi)

        target_next_obs = aux_data.get('next_obs', None)
        if pred_next_obs is not None and target_next_obs is not None:
            pred_loss_dict: dict[str, jax.Array] = {}
            for obs_key in pred_next_obs.keys():
                if obs_key not in target_next_obs:
                    continue
                sq_err = jnp.square(pred_next_obs[obs_key] - target_next_obs[obs_key]).reshape(*pred_next_obs[obs_key].shape[:2], -1).mean(-1)
                if 'mask' in aux_data:
                    key_loss = (sq_err * aux_data['mask']).mean()
                else:
                    key_loss = sq_err.mean()
                pred_loss_dict[obs_key] = key_loss
                info[f'pred_loss[{obs_key}]'] = key_loss
            if len(pred_loss_dict) > 0:
                info['pred_loss'] = jnp.mean(jnp.stack(list(pred_loss_dict.values()), axis=0))

        return log_p, entropy, val, info

    def ppo_loss(
        self,
        model_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[float, dict[str, float]]:
        loss, info = super().ppo_loss(model_state, buffer, advantage, target_val, aux_data, rng)
        pred_loss = info.get('pred_loss', None)
        if pred_loss is not None:
            loss = loss + self.ppo_config['pred_loss_coef'] * pred_loss
            info['pred_loss_term'] = self.ppo_config['pred_loss_coef'] * pred_loss
        info['loss'] = loss
        return loss, info

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
                - agent_state: The agent's recurrent state (e.g., RNN hidden states with shape [batch_size, minibatch_seq_len, *rnn_state_shape]).

        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        model_state = self.model_state_from_trainer_state(trainer_state)
        (loss, info), grads = jax.value_and_grad(self.ppo_loss, has_aux=True)(
            model_state,
            buffer=buffer,
            advantage=advantage,
            target_val=target_val,
            aux_data=aux_data,
            rng=rng
        )

        info['grad_norm'] = global_norm(grads['actor_critic_params'])
        new_actor_critic_train_state = trainer_state['actor_critic_train_state'].apply_gradients(grads=grads['actor_critic_params'])
        return {'actor_critic_train_state': new_actor_critic_train_state}, info
