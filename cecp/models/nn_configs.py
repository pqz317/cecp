from abc import abstractmethod
from typing import Sequence

import flax.linen as nn
import flax.struct as struct
from flax.struct import PyTreeNode

from .nn_blocks import Activation, CNN, CNNTranspose, MLP


class NNConfig(PyTreeNode):
    @abstractmethod
    def create(self) -> nn.Module:
        ...

class CNNConfig(NNConfig):
    feature_lst: Sequence[int] = struct.field(default_factory=lambda: [64, 64, 64])
    kernel_lst: Sequence[int | Sequence[int]] = struct.field(default_factory=lambda: [(3, 3), (3, 3), (3, 3)])
    stride_lst: Sequence[int | Sequence[int]] = struct.field(default_factory=lambda: [1, 1, 1])
    activation_type: Activation.Type = 'relu'
    post_mlp_hidden_size: int = 64
    post_mlp_hidden_size_lst: Sequence[int] | None = None
    post_mlp_activation_type: Activation.Type = 'relu'
    kernel_init_scale: float | None = None
    post_mlp_kernel_init_scale: float | None = None
    
    def create(self) -> nn.Module:
        return nn.Sequential([
            CNN(
                feature_lst=self.feature_lst,
                kernel_lst=self.kernel_lst,
                stride_lst=self.stride_lst,
                activation_type=self.activation_type,
                kernel_init_scale=self.kernel_init_scale,
            ),
            lambda x: x.reshape(*x.shape[:-3], -1),
            MLP(
                hidden_size_lst=(
                    self.post_mlp_hidden_size_lst
                    if self.post_mlp_hidden_size_lst is not None
                    else [self.post_mlp_hidden_size]
                ),
                activation_type=self.post_mlp_activation_type,
                kernel_init_scale=self.post_mlp_kernel_init_scale,
            )
        ])


class CNNTransposeConfig(NNConfig):
    feature_lst: Sequence[int] = struct.field(default_factory=lambda: [64, 64, 3])
    kernel_lst: Sequence[int | Sequence[int]] = struct.field(default_factory=lambda: [(3, 3), (3, 3), (3, 3)])
    stride_lst: Sequence[int | Sequence[int]] = struct.field(default_factory=lambda: [1, 1, 1])
    activation_type: Activation.Type = 'relu'
    use_final_activation: bool = False

    def create(self) -> nn.Module:
        return CNNTranspose(
            feature_lst=self.feature_lst,
            kernel_lst=self.kernel_lst,
            stride_lst=self.stride_lst,
            activation_type=self.activation_type,
            use_final_activation=self.use_final_activation
        )

    @classmethod
    def from_cnn_config(cls, cnn_config: CNNConfig, output_channels: int, use_final_activation: bool = False) -> 'CNNTransposeConfig':
        normalized_stride_lst = []
        for kernel, stride in zip(cnn_config.kernel_lst, cnn_config.stride_lst):
            if isinstance(stride, int):
                if isinstance(kernel, int):
                    normalized_stride_lst.append((stride,))
                else:
                    normalized_stride_lst.append(tuple([stride for _ in range(len(kernel))]))
            else:
                normalized_stride_lst.append(tuple(stride))

        return cls(
            feature_lst=list(reversed(cnn_config.feature_lst[:-1])) + [output_channels],
            kernel_lst=list(reversed(cnn_config.kernel_lst)),
            stride_lst=list(reversed(normalized_stride_lst)),
            activation_type=cnn_config.activation_type,
            use_final_activation=use_final_activation
        )

class MLPConfig(NNConfig):
    hidden_layers: int = 2
    hidden_size: int = 64
    hidden_size_lst: Sequence[int] | None = None
    activation_type: Activation.Type = 'relu'
    kernel_init_scale: float | None = None

    def create(self) -> nn.Module:
        return MLP(
            hidden_size_lst=(
                self.hidden_size_lst
                if self.hidden_size_lst is not None
                else [self.hidden_size for _ in range(self.hidden_layers)]
            ),
            activation_type=self.activation_type,
            kernel_init_scale=self.kernel_init_scale,
        )
