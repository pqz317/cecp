from typing import Sequence

import flax.linen as nn
import jax

from .activations import Activation


class CNNTranspose(nn.Module):
    feature_lst: Sequence[int]
    kernel_lst: Sequence[int | Sequence[int]]
    stride_lst: Sequence[int | Sequence[int]]
    activation_type: Activation.Type = 'relu'
    use_final_activation: bool = False

    def setup(self):
        self.cnn_layers = [
            nn.ConvTranspose(
                features=num_features,
                kernel_size=kernel_shape,
                strides=stride
            )
            for (num_features, kernel_shape, stride) in zip(self.feature_lst, self.kernel_lst, self.stride_lst)
        ]
        self.activation_fn = Activation.get(self.activation_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, cnn_layer in enumerate(self.cnn_layers):
            x = cnn_layer(x)
            if i < len(self.cnn_layers) - 1 or self.use_final_activation:
                x = self.activation_fn(x)
        return x