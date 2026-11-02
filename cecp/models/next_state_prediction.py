from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..envs.spaces import Action, ActionSpace, ContinuousSpace, DiscreteSpace, Observation, ObservationSpace
from .nn_blocks import Activation, MLP
from .nn_configs import CNNConfig, CNNTransposeConfig, MLPConfig, NNConfig


class MLPObservationDecoder(nn.Module):
    """Decodes a feature vector into an observation using an MLP."""
    target_shape: Sequence[int]
    hidden_size_lst: Sequence[int]
    activation_type: Activation.Type = 'relu'

    def setup(self) -> None:
        if len(self.hidden_size_lst) > 0:
            self.mlp = MLP(
                hidden_size_lst=self.hidden_size_lst,
                activation_type=self.activation_type
            )
        else:
            self.mlp = None
        self.output_layer = nn.Dense(int(np.prod(self.target_shape)))

    def __call__(self, feature: jax.Array) -> jax.Array:
        x = feature
        if self.mlp is not None:
            x = self.mlp(x)
        x = self.output_layer(x)
        return x.reshape(*x.shape[:-1], *self.target_shape)


class CNNObservationDecoder(nn.Module):
    """
    Decodes a feature vector into an observation using a CNN-based decoder."""
    target_shape: Sequence[int]
    start_channels: int
    decoder_config: CNNTransposeConfig

    def setup(self) -> None:
        h, w, _ = self.target_shape
        self.start_h = h
        self.start_w = w
        self.proj = nn.Dense(self.start_h * self.start_w * self.start_channels)
        self.decoder = self.decoder_config.create()

    def __call__(self, feature: jax.Array) -> jax.Array:
        x = self.proj(feature)
        x = x.reshape(*x.shape[:-1], self.start_h, self.start_w, self.start_channels)
        return self.decoder(x)


class ObservationPredictor(nn.Module):
    obs_space: ObservationSpace
    embedding_extractor_config: dict[str, NNConfig]

    def setup(self) -> None:
        decoders = {}
        for key in sorted(self.embedding_extractor_config.keys()):
            if key not in self.obs_space:
                continue
            obs_shape = self.obs_space[key].shape
            encoder_config = self.embedding_extractor_config[key]
            if isinstance(encoder_config, MLPConfig):
                hidden_size_lst = (
                    list(encoder_config.hidden_size_lst)
                    if encoder_config.hidden_size_lst is not None
                    else [encoder_config.hidden_size for _ in range(encoder_config.hidden_layers)]
                )
                decoders[key] = MLPObservationDecoder(
                    target_shape=obs_shape,
                    hidden_size_lst=list(reversed(hidden_size_lst)),
                    activation_type=encoder_config.activation_type
                )
            elif isinstance(encoder_config, CNNConfig):
                decoder_config = CNNTransposeConfig.from_cnn_config(
                    encoder_config,
                    output_channels=obs_shape[-1],
                    use_final_activation=False
                )
                decoders[key] = CNNObservationDecoder(
                    target_shape=obs_shape,
                    start_channels=encoder_config.feature_lst[-1],
                    decoder_config=decoder_config
                )
            else:
                raise NotImplementedError(f"Decoder not implemented for encoder config type {type(encoder_config)} at key '{key}'.")
        self.decoders = decoders

    def __call__(self, feature: jax.Array, action_feature: jax.Array | None = None) -> Observation:
        predictor_input = feature if action_feature is None else jnp.concatenate([feature, action_feature], axis=-1)
        return Observation({
            key: self.decoders[key](predictor_input)
            for key in sorted(self.decoders.keys())
        })


class ActionFeatureEncoder(nn.Module):
    action_space: ActionSpace

    def __call__(self, action: Action) -> jax.Array:
        action_features = []
        for action_name in sorted(self.action_space.keys()):
            if action_name not in action:
                continue
            action_value = action[action_name]
            space = self.action_space[action_name]
            if isinstance(space, DiscreteSpace):
                action_feature = jax.nn.one_hot(action_value.astype(jnp.int32), space.n, dtype=jnp.float32)
            elif isinstance(space, ContinuousSpace):
                action_feature = action_value.astype(jnp.float32)
                action_feature = action_feature.reshape(*action_feature.shape[:-len(space.shape)], -1)
            else:
                raise NotImplementedError(
                    f"Action encoder only supports DiscreteSpace and ContinuousSpace, got {type(space)} for action[{action_name}]"
                )
            action_features.append(action_feature)

        if len(action_features) == 0:
            raise ValueError("ActionFeatureEncoder received no matching action keys for the configured action_space.")

        return jnp.concatenate(action_features, axis=-1)