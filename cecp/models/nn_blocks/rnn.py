from functools import partial
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from ...utils import split_rng_to_list


class LayerNormGRUCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, h: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dense = partial(nn.Dense, features=self.features, use_bias=False, kernel_init=nn.initializers.orthogonal())
        ln_ir, ln_hr, ln_iz, ln_hz, ln_in, ln_hn = nn.LayerNorm(), nn.LayerNorm(), nn.LayerNorm(), nn.LayerNorm(), nn.LayerNorm(), nn.LayerNorm()

        r = nn.sigmoid(ln_ir(dense(name='ir')(x)) + ln_hr(dense(name='hr')(h))) # Update gate
        z = nn.sigmoid(ln_iz(dense(name='iz')(x)) + ln_hz(dense(name='hz')(h))) # Reset gate
        n = nn.tanh(ln_in(dense(name='in')(x)) + r * ln_hn(dense(name='hn')(h))) # Candidate
        new_h = (1. - z) * n + z * h
        return new_h, new_h

class MultiLayerGRU(nn.Module):
    rnn_layers: int
    hidden_size: int
    use_layer_norm: bool
    """
    Applies the GRU unit layer-by-layer.
    
    Architecture:
        This module stacks multiple GRU layers sequentially:
            x_0 -> GRU_1 -> x_1 -> GRU_2 -> x_2 -> ... x_{L-1} -> GRU_L -> x_L
        The carry is a concatenation of all layer states:
            carry = [h_1, h_2, ..., h_L]
        Only the top-layer output x_L is returned as the representation.

    [Input]
        - carry: Recurrent state before processing the step: [h_1, h_2, ..., h_L].
            - The shape of h_i: [*batch_shape, hidden_size].
        - x: Input to the multi-layer GRU, with shape [*batch_shape, input_dim].

    [Output]
        - carry: Updated recurrent state after processing the step: [h_1', h_2', ..., h_L'].
            - The shape of h_i': [*batch_shape, hidden_size].
        - x: Output (representation layer), with shape [*batch_shape, hidden_size].
    """

    def setup(self):
        self.gru_layers = [
            (LayerNormGRUCell if self.use_layer_norm else nn.GRUCell)(features=self.hidden_size)
            for _ in range(self.rnn_layers)
        ]

    def __call__(self, carry: Sequence[jax.Array], x: jax.Array,) -> tuple[list[jax.Array], jax.Array]:
        new_carry = []
        for h, gru_cell in zip(carry, self.gru_layers):
            h, x = gru_cell(h, x)
            new_carry.append(h)
        return new_carry, x
    
    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, batch_shape: Sequence[int] = ()) -> list[jax.Array]:
        rng, rng_carry_lst = split_rng_to_list(rng, self.rnn_layers)
        return [
            nn.GRUCell(features=self.hidden_size).initialize_carry(rng_carry, input_shape=(*batch_shape, 1)) # dummy observation with feature_dim=1
            for rng_carry in rng_carry_lst
        ]


class MultiLayerLSTM(nn.Module):
    rnn_layers: int
    hidden_size: int

    def setup(self):
        self.lstm_layers = [
            nn.OptimizedLSTMCell(features=self.hidden_size)
            for _ in range(self.rnn_layers)
        ]

    def __call__(self, carry: Sequence[tuple[jax.Array, jax.Array]], x: jax.Array) -> tuple[list[tuple[jax.Array, jax.Array]], jax.Array]:
        new_carry = []
        for layer_carry, lstm_cell in zip(carry, self.lstm_layers):
            layer_carry, x = lstm_cell(layer_carry, x)
            new_carry.append(layer_carry)
        return new_carry, x

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, batch_shape: Sequence[int] = ()) -> list[tuple[jax.Array, jax.Array]]:
        rng, rng_carry_lst = split_rng_to_list(rng, self.rnn_layers)
        return [
            nn.OptimizedLSTMCell(features=self.hidden_size).initialize_carry(rng_carry, input_shape=(*batch_shape, 1))
            for rng_carry in rng_carry_lst
        ]

class RNN(nn.Module):
    rnn_layers: int = 1
    hidden_size: int = 64
    use_layer_norm: bool = False
    time_major: bool = False
    cell_type: Literal['gru', 'lstm'] = 'gru'

    def setup(self):
        if self.cell_type not in ('gru', 'lstm'):
            raise ValueError(f"Unsupported cell_type={self.cell_type}. Expected one of ['gru', 'lstm']")
        scanned_cell = nn.scan(
            MultiLayerGRU if self.cell_type == 'gru' else MultiLayerLSTM,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0 if self.time_major else 1,
            out_axes=0 if self.time_major else 1
        )
        if self.cell_type == 'gru':
            self.scan_gru = scanned_cell(
                rnn_layers=self.rnn_layers,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm,
            )
        else:
            self.scan_gru = scanned_cell(
                rnn_layers=self.rnn_layers,
                hidden_size=self.hidden_size,
            )
        
    def __call__(self, carry: dict[str, Sequence[jax.Array]], x: jax.Array) -> tuple[dict[str, list[jax.Array]], jax.Array]:
        """
        Applies the recurrent model over the input sequence.

        [Input]
            - carry: Recurrent state [h_1, h_2, ..., h_L] before processing the sequence, with shape [batch_size, hidden_size] for each unit.
            - x: Input sequence to the RNN, with shape [batch_size, sequence_length, input_dim] or [batch_size, input_dim] (i.e., sequence_length=1).

        [Output]
            - carry: Updated recurrent state [h_1', h_2', ..., h_L'] after processing the sequence, with shape [batch_size, hidden_size] for each unit.
            - x: Output sequence (representation layer), with shape [batch_size, sequence_length, hidden_size] or [batch_size, input_dim].
        """
        carry = carry['rnn_carry']
        if len(x.shape) == 2:
            x = jnp.expand_dims(x, axis=1)
            carry, x = self.scan_gru(carry, x)
            x = jnp.squeeze(x, axis=1)
        elif len(x.shape) == 3:
            carry, x = self.scan_gru(carry, x)
        else:
            raise ValueError(
                f"RNN expects input x with shape [batch_size, sequence_length, hidden_size] or [batch_size, input_dim], "
                f"but got shape {x.shape}."
            )
        return {'rnn_carry': carry}, x

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, batch_size: int = 1) -> list[jax.Array]:
        if self.cell_type == 'gru':
            rnn_cell = MultiLayerGRU(
                rnn_layers=self.rnn_layers,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm
            )
        elif self.cell_type == 'lstm':
            rnn_cell = MultiLayerLSTM(
                rnn_layers=self.rnn_layers,
                hidden_size=self.hidden_size,
            )
        else:
            raise ValueError(f"Unsupported cell_type={self.cell_type}. Expected one of ['gru', 'lstm']")
        return {'rnn_carry': rnn_cell.initialize_carry(rng, batch_shape=(batch_size,))}
