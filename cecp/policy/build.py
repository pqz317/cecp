from ..config import ArchitectureConfig, PredictiveConfig
from ..envs import ActionSpace, ObservationSpace
from .actor_critic import PredictiveSharedActorCriticModel, SharedActorCriticModel, UnifiedSharedActorCriticModel


def build_actor_critic(
    arch: ArchitectureConfig,
    predictive: PredictiveConfig,
    obs_space: ObservationSpace,
    action_space: ActionSpace,
    output_nn_depth: int | None = -1,
) -> UnifiedSharedActorCriticModel:
    """Build the CEC (no prediction head) or CECP (prediction head) actor-critic from config.

    `output_nn_depth != -1` prints a Flax `tabulate` summary of the network at that depth.
    """
    kwargs = dict(
        cnn_feature_lst=tuple(arch.encoder_conv_sizes),
        cnn_kernel_lst=tuple((k, k) for k in arch.encoder_conv_kernel_sizes),
        cnn_stride_lst=tuple(arch.encoder_conv_strides),
        encoder_fc_hidden_size_lst=tuple(arch.encoder_fc_sizes),
        use_rnn=arch.use_rnn,
        rnn_hidden_layers=arch.rnn_layers,
        rnn_hidden_size=arch.rnn_hidden_size,
        rnn_cell_type=arch.rnn_type,
        actor_head_hidden_size_lst=tuple(arch.actor_head_sizes),
        critic_head_hidden_size_lst=tuple(arch.critic_head_sizes),
        encoder_kernel_init_scale=arch.encoder_kernel_init_scale,
        head_kernel_init_scale=arch.head_kernel_init_scale,
        actor_output_kernel_init_scale=arch.actor_output_kernel_init_scale,
        critic_output_kernel_init_scale=arch.critic_output_kernel_init_scale,
        readout_size=arch.readout_size,
        activation_type=arch.activation_type,
        output_nn_depth=output_nn_depth,
    )
    if predictive.use_predictive_objective:
        net = PredictiveSharedActorCriticModel.build_configurable_predictive_shared_actor_critic(obs_space, action_space, **kwargs)
        return PredictiveSharedActorCriticModel(net)
    net = SharedActorCriticModel.build_configurable_shared_actor_critic(obs_space, action_space, **kwargs)
    return SharedActorCriticModel(net)
