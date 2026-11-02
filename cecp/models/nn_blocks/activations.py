from typing import Callable, Literal

import flax.linen as nn
import jax


class Activation:
    Type = Literal[
        'relu', 'tanh', 'sigmoid', 'swish', 'gelu', 'leaky_relu'
    ]

    table = {
        'relu': nn.relu,
        'tanh': nn.tanh,
        'sigmoid': nn.sigmoid,
        'swish': nn.swish,
        'gelu': nn.gelu,
        'leaky_relu': nn.leaky_relu,
    }

    @staticmethod
    def get(name: Type) -> Callable[[jax.Array], jax.Array]:
        return Activation.table[name]
