from typing import Sequence

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode


def partition_roles_equal(num_roles: int, num_shards: int) -> list[list[int]]:
    if num_roles < 1:
        raise ValueError(f"num_roles must be >= 1, got {num_roles}")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if num_shards > num_roles:
        raise ValueError(
            f"num_shards ({num_shards}) cannot exceed num_roles ({num_roles}) when equal sharding is required"
        )
    if num_roles % num_shards != 0:
        raise ValueError(
            f"equal sharding requires num_roles % num_shards == 0, got {num_roles} % {num_shards}"
        )
    roles_per_shard = num_roles // num_shards
    role_indices = list(range(num_roles))
    return [
        role_indices[shard_idx * roles_per_shard:(shard_idx + 1) * roles_per_shard]
        for shard_idx in range(num_shards)
    ]


def select_training_devices(preferred_kind: str = 'gpu') -> list[jax.Device]:
    try:
        devices = list(jax.devices(preferred_kind))
    except RuntimeError:
        devices = []
    if len(devices) == 0:
        devices = list(jax.devices())
    return devices


def build_role_to_shard_local(role_shards: Sequence[Sequence[int]]) -> dict[int, tuple[int, int]]:
    return {
        role_id: (shard_idx, local_idx)
        for shard_idx, role_ids in enumerate(role_shards)
        for local_idx, role_id in enumerate(role_ids)
    }


def pytree_repeat_stack(node: PyTreeNode, batch_shape: Sequence[int]):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (*batch_shape, *(x.shape if hasattr(x, 'shape') else ()))),
        node
    )

def split_rng_to_list(rng: jax.Array, batch_size: int = 1) -> tuple[jax.Array, list[jax.Array]]:
    rng, rng_subkey = jax.random.split(rng)
    rng_lst = list(jax.random.split(rng_subkey, batch_size))
    return rng, rng_lst

def split_into_minibatch(rng: jax.Array, data: PyTreeNode, minibatch_seq_len: int, num_minibatches: int | None = 1, minibatch_num_chunks: int | None = None) -> PyTreeNode:
    """
    [Input]
        - data: [seq_length, batch_size, *data_attribute_shape]
    
    [Output]
        - minibatches: [num_minibatches, minibatch_num_chunks, minibatch_seq_len, *data_attribute_shape]
    
    [Notation]
        - L: seq_length
        - L': minibatch_seq_len
        - B: batch_size (number of sequences)
        - B': num_chunks_per_seq
        - A: *attribute_shape
        - N: num_minibatches (total number of chunks from all sequences)
    """
    def reshape_data(data: jax.Array):
        # data: [L, B, A]
        seq_length = data.shape[0]
        chunks_per_episode = seq_length // minibatch_seq_len
        data = data[:chunks_per_episode*minibatch_seq_len].reshape(chunks_per_episode, minibatch_seq_len, *data.shape[1:]) # [B', L', B, A]
        data = jnp.swapaxes(data, 1, 2).reshape(-1, minibatch_seq_len, *data.shape[3:]) # [BB', L', A]
        p = jax.random.permutation(rng, data.shape[0])
        data = jnp.take(data, p, axis=0)
        n = data.shape[0] // minibatch_num_chunks if minibatch_num_chunks is not None else num_minibatches
        data = data[:data.shape[0]//n*n].reshape(n, -1, *data.shape[1:]) # [N, BB'/N, L, A]
        return data # [N, BB'/N, L', A] 
    return jax.tree_util.tree_map(reshape_data, data)

def global_norm(params: PyTreeNode) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(params)))
