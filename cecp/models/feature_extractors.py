from typing import Literal

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode

from ..envs.spaces import Observation
from .nn_blocks import Activation, MLP, RNN
from .nn_configs import NNConfig


class EmbeddingExtractor(nn.Module):
    """
    Extracts per-modality embeddings from a structured observation.

    This module applies a set of feature extractors (CNN / MLP / etc.)
    to different keys in the observation dictionary independently.

    Each observation key corresponds to one neural network defined
    by `nn_configs[key]`. The outputs are concatenated later by the
    downstream module.

    Architecture:
        obs (dict[str, Array])
            ├─ key_1 → FeatureNet_1 → embedding_1
            ├─ key_2 → FeatureNet_2 → embedding_2
            └─ ...
        (optional) LayerNorm is applied per embedding.
    """
    nn_configs: dict[str, NNConfig]
    use_embedding_layer_norm: bool = True

    def setup(self) -> None:
        self.feature_extractors = {
            k: self.nn_configs[k].create()
            for k in sorted(self.nn_configs.keys())
        }
        if self.use_embedding_layer_norm:
            self.layer_norms = {
                k: nn.LayerNorm() for k in sorted(self.nn_configs.keys())
            }
    
    def __call__(self, obs: Observation) -> dict[str, jax.Array]:
        embedding_dict = {
            k: self.feature_extractors[k](obs[k])
            for k in sorted(set(obs.keys()) & set(self.feature_extractors.keys()))
        }
        if self.use_embedding_layer_norm:
            embedding_dict = {k: self.layer_norms[k](v) for k, v in embedding_dict.items()}
        return embedding_dict

class RecurrentFeatureExtractor(nn.Module):
    """
    Extracts a unified feature representation from structured observations,
    optionally with temporal modeling via an RNN.

    This module is composed of four logical stages:

    1. Embedding extraction (per observation key)
       - Each observation component is encoded independently by
         `EmbeddingExtractor`.

    2. Feature aggregation
       - All embeddings are concatenated along the feature dimension.
       - An optional MLP is applied to mix information across modalities.

    3. Temporal modeling (optional)
       - An RNN processes the aggregated features over time, maintaining
         a recurrent hidden state.

    4. Output normalization (optional)
       - A final LayerNorm can be applied to the output features.
    """
    embedding_extractor_config: dict[str, NNConfig]
    use_embedding_layer_norm: bool = True
    aggr_mlp_layers: int = 0
    aggr_mlp_hidden_size: int = 64
    aggr_mlp_activation_type: Activation.Type = 'relu'
    use_rnn: bool = False
    rnn_hidden_layers: int = 1
    rnn_hidden_size: int = 64
    rnn_cell_type: Literal['gru', 'lstm'] = 'gru'
    use_rnn_layer_norm: bool = True
    use_final_layer_norm: bool = False
    
    def setup(self) -> None:
        self.embedding_extractor = EmbeddingExtractor(
            nn_configs=self.embedding_extractor_config,
            use_embedding_layer_norm=self.use_embedding_layer_norm
        )
        if self.aggr_mlp_layers > 0:
            self.aggr_mlp = MLP(
                hidden_size_lst=[self.aggr_mlp_hidden_size for i in range(self.aggr_mlp_layers)],
                activation_type=self.aggr_mlp_activation_type
            )
        if self.use_rnn:
            self.rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm,
                cell_type=self.rnn_cell_type,
            )
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm()

    def __call__(self, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        """
        [Input]
            - obs[key]: [batch_size, sequence_length, ...].
            - rnn_state: PyTree compatible with the RNN implementation.

        [Output]
            - feature: [batch_size, sequence_length, feature_dim].
            - rnn_state: updated recurrent state.
        """
        embedding_dict = self.embedding_extractor(obs)
        feature = jnp.concatenate([embedding_dict[k] for k in sorted(embedding_dict.keys())], axis=-1)
        if self.aggr_mlp_layers > 0:
            feature = self.aggr_mlp(feature)
        if self.use_rnn:
            rnn_state, feature = self.rnn_model(rnn_state, feature)
        if self.use_final_layer_norm:
            feature = self.final_layer_norm(feature)
        return rnn_state, feature

    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        if self.use_rnn:
            rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm,
                cell_type=self.rnn_cell_type,
            )
            return rnn_model.initialize_carry(rng, batch_size)
        else:
            return jnp.zeros((batch_size,))
