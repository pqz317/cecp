from typing import Sequence

import jax
import flax.linen as nn

from .activations import Activation


class MLP(nn.Module):
    hidden_size_lst: Sequence[int]
    activation_type: Activation.Type = 'relu'
    kernel_init_scale: float | None = None
    """
    Applies a stack of fully-connected (Dense) layers.

    Architecture:
        This module applies a sequence of Dense layers with the same activation
        function after each layer:
            x -> Dense_1 -> activation_1 -> x_1 -> Dense_2 -> activation_2 -> x_2 -> ... -> x_{L-1} -> Dense_L activation_L -> x_L

    [Input]
        - x: Input tensor with shape [*batch_shape, input_dim].

    [Output]
        - x_L: Output tensor of the final Dense layer, with shape
              [*batch_shape, hidden_size_lst[-1]].
    """

    def setup(self):
        kernel_init = nn.initializers.orthogonal(self.kernel_init_scale) if self.kernel_init_scale is not None else nn.initializers.lecun_normal()
        self.fc_layers = [nn.Dense(hidden_size, kernel_init=kernel_init) for hidden_size in self.hidden_size_lst]
        self.activation_fn = Activation.get(self.activation_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        [Input]
            - x: [*batch_shape, input_dim]
        
        [Output]
            - x: Finaly layer with shape [*batch_shape, hidden_size_lst[-1]]
        """
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = self.activation_fn(x)
        return x