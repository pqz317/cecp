from typing import Sequence

import jax
import flax.linen as nn

from .activations import Activation


class CNN(nn.Module):
    feature_lst: Sequence[int]
    kernel_lst: Sequence[int | Sequence[int]]
    stride_lst: Sequence[int | Sequence[int]]
    activation_type: Activation.Type = 'relu'
    kernel_init_scale: float | None = None
    """
    Applies a stack of convolutional layers.

    Architecture:
        This module applies multiple convolutional layers sequentially,
        each followed by the same activation function:
            x -> Conv_1 -> activation_1 -> x_1 -> Conv_2 -> activation_2 -> x_2 ... x_{L-1} -> Conv_L -> activation_L -> x_L

    [Input]
        - x: Input tensor with shape [*batch_shape, W, H, C],
             where C is the number of input channels.

    [Output]
        - x_L: Output tensor of the final convolutional layer, with shape
              [*batch_shape, W', H', feature_lst[-1]].
    """
    def setup(self):
        kernel_init = nn.initializers.orthogonal(self.kernel_init_scale) if self.kernel_init_scale is not None else nn.initializers.lecun_normal()
        self.cnn_layers = [
            nn.Conv(
                features=num_features,
                kernel_size=kernel_shape,
                strides=stride,
                kernel_init=kernel_init,
            )
            for (num_features, kernel_shape, stride) in zip(self.feature_lst, self.kernel_lst, self.stride_lst)
        ]
        self.activation_fn = Activation.get(self.activation_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        [Input]
            - x: [*batch_shape, W, H, C]
        
        [Output]
            - x: Final layer with shape[*batch_shape, W', H', feature_lst[-1]].
        """
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            x = self.activation_fn(x)
        return x
