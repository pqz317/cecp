"""Config dataclasses shared by training, model rebuilding and evaluation.

Training-loop configs (PPO, sharding, partner pool, ...) live with the training scripts; the ones here
are what `cecp.runs.load_policy` needs to rebuild an environment and a model from a run's config.yaml.
"""
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeVar


@dataclass
class CoopForagingEnvConfig:
    name: Literal['coop_foraging'] = 'coop_foraging'
    reward_shaping_steps: int = int(1e10)
    heldout_goal_ratio: float = 0.0
    heldout_goal_seed: int = 0
    eval_on_reserved_goals: bool = False
    obstacle_density: float = 0.1
    train_layout_mode: Literal['random', 'fixed_procgen'] = 'random'
    train_procgen_seed: int = 0


@dataclass
class OvercookedEnvConfig:
    name: Literal['overcooked'] = 'overcooked'
    layout: str = 'cramped_room'
    heldout_layout_seed: int = 0
    heldout_layout_num_samples: int = 100
    eval_on_reserved_layouts: bool = False
    pad_obs_shape_to: tuple[int, ...] | None = None
    train_shaped_reward_factor: float = 0.0
    reward_shaping_steps: int = int(1e10)
    train_layout_mode: str = 'random'
    train_procgen_seed: int = 0


EnvConfig = CoopForagingEnvConfig | OvercookedEnvConfig


@dataclass
class ArchitectureConfig:
    encoder_conv_sizes: list[int] = field(default_factory=lambda: [64, 32])
    encoder_conv_kernel_sizes: list[int] = field(default_factory=lambda: [2, 2])
    encoder_conv_strides: list[int] = field(default_factory=lambda: [1, 1])
    encoder_fc_sizes: list[int] = field(default_factory=lambda: [512, 512])
    use_rnn: bool = True
    rnn_type: str = 'lstm'
    rnn_layers: int = 1
    rnn_hidden_size: int = 256
    actor_head_sizes: list[int] = field(default_factory=lambda: [256, 192, 128, 64])
    critic_head_sizes: list[int] = field(default_factory=lambda: [512, 256, 192, 128])
    encoder_kernel_init_scale: float = math.sqrt(2.0)
    head_kernel_init_scale: float = 2.0
    actor_output_kernel_init_scale: float = 0.01
    critic_output_kernel_init_scale: float = 1.0
    readout_size: int = 256
    # Activation function used throughout the encoder and heads.
    # One of: 'relu', 'tanh', 'sigmoid', 'swish', 'gelu', 'leaky_relu'.
    activation_type: str = 'relu'

    def __post_init__(self) -> None:
        n_conv = len(self.encoder_conv_sizes)
        if len(self.encoder_conv_kernel_sizes) != n_conv:
            raise ValueError(
                f"len(encoder_conv_kernel_sizes)={len(self.encoder_conv_kernel_sizes)} must match len(encoder_conv_sizes)={n_conv}"
            )
        if len(self.encoder_conv_strides) != n_conv:
            raise ValueError(
                f"len(encoder_conv_strides)={len(self.encoder_conv_strides)} must match len(encoder_conv_sizes)={n_conv}"
            )
        if self.use_rnn:
            if self.rnn_type not in ('gru', 'lstm'):
                raise ValueError(f"architecture.rnn_type must be one of ('gru', 'lstm'), got {self.rnn_type}")
            if self.rnn_layers < 1:
                raise ValueError(f"architecture.rnn_layers must be >= 1, got {self.rnn_layers}")
            if self.rnn_hidden_size < 1:
                raise ValueError(f"architecture.rnn_hidden_size must be >= 1, got {self.rnn_hidden_size}")


@dataclass
class PredictiveConfig:
    use_predictive_objective: bool = False
    pred_loss_coef: float = 1.0


T = TypeVar('T')


def from_mapping(cls: type[T], mapping: Mapping[str, Any]) -> T:
    """Build dataclass `cls` from a plain mapping (e.g. a section of a run's config.yaml).

    Unknown keys are an error rather than silently ignored, so a renamed field cannot quietly fall back
    to its default when a run is reloaded.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(mapping) - names
    if unknown:
        raise ValueError(f'{cls.__name__} got unknown keys {sorted(unknown)}')
    kwargs = dict(mapping)
    if kwargs.get('pad_obs_shape_to') is not None:
        kwargs['pad_obs_shape_to'] = tuple(kwargs['pad_obs_shape_to'])
    return cls(**kwargs)


def env_config_from_mapping(mapping: Mapping[str, Any]) -> EnvConfig:
    name = mapping.get('name')
    if name == 'coop_foraging':
        return from_mapping(CoopForagingEnvConfig, mapping)
    if name == 'overcooked':
        return from_mapping(OvercookedEnvConfig, mapping)
    raise ValueError(f"env.name must be one of ('coop_foraging', 'overcooked'), got {name!r}")
