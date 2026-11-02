"""Adapter mapping native human-demo Overcooked observations into the `<layout>_9`
env observation frame, so behavioral-cloning data recorded on a native-size layout
(e.g. ``counter_circuit`` = ``[5, 8, 26]``) can train a model that is initialized
against the 9x9 env obs space (``[9, 9, 26]``).

jaxmarl's Overcooked hardcodes ``height == width == 9``, so only the ``*_9`` layouts
render correctly. Each ``<layout>_9`` embeds its native grid at the **top-left,
un-rotated** on an all-wall 9x9 base (``layouts.make_9x9_layout`` with ``rotate=False``).
This adapter therefore:

1. places the demo's DYNAMIC channels into the 9x9 frame at the layout's embedding
   offset (identity top-left for four layouts; a one-row top crop for ``asymm_advantages``,
   whose ``_9`` variant drops the native top wall-border row);
2. overwrites the STATIC layout channels (pot / wall / onion-pile / plate-pile / goal)
   across the full 9x9 with the target ``_9`` env's canonical layout — so wherever the
   native layout and the ``_9`` layout disagree (relocated dispensers, duplicated corner
   tiles, wall padding beyond the native region) the ``_9`` env wins; and
3. broadcasts the spatially-uniform urgency channel across the full 9x9.

Observation channel semantics (jaxmarl ``Overcooked.get_obs``), ``[H, W, 26]``:
    0-9   agent position / orientation one-hots      (dynamic)
    10    pot locations                              (static)
    11    wall / counter locations                   (static)
    12    onion-pile locations                       (static)
    13    tomato-pile (always 0)
    14    plate-pile locations                       (static)
    15    goal / delivery locations                  (static)
    16    onions in pot                              (dynamic)
    17    tomatoes in pot (always 0)
    18    onions in soup / held dish                 (dynamic)
    19    tomatoes in soup (always 0)
    20    pot cook timer                             (dynamic)
    21    soup ready / held dish                     (dynamic)
    22    loose / held plate locations               (dynamic)
    23    loose / held onion locations               (dynamic)
    24    loose tomatoes (always 0)
    25    urgency (spatially uniform whole-grid flag)
"""

from typing import Sequence

import jax
import jax.numpy as jnp


# Static layout channels overwritten from the target `_9` env (per the "default to
# the _9 layout on mismatch" rule). Channels 13/17/19/24 are always 0, so the zero-init
# already matches; they need no overwrite.
_STATIC_CHANNELS: tuple[int, ...] = (10, 11, 12, 14, 15)

# Uniform whole-grid urgency channel — must cover walls / padding, not just the native region.
_URGENCY_CHANNEL: int = 25

# How each native (dataset) layout embeds into its `<layout>_9` 9x9 frame:
#   (src_row0, src_col0, n_rows, n_cols, dst_row0, dst_col0)
# All four non-asymm layouts are a pure top-left, un-rotated embedding. asymm_advantages
# is native 5x9 but its `_9` content is 4x9 with the native top wall-border row dropped,
# so its native rows 1..4 map to `_9` rows 0..3.
_NATIVE_TO_9_PLACEMENT: dict[str, tuple[int, int, int, int, int, int]] = {
    "cramped_room":     (0, 0, 4, 5, 0, 0),
    "coord_ring":       (0, 0, 5, 5, 0, 0),
    "forced_coord":     (0, 0, 5, 5, 0, 0),
    "counter_circuit":  (0, 0, 5, 8, 0, 0),
    "asymm_advantages": (1, 0, 4, 9, 0, 0),
}


class OvercookedObsAdapter:
    """Callable that lifts native demo ``grid_2d`` ``[..., h, w, 26]`` into the target
    ``<layout>_9`` env frame ``[..., 9, 9, 26]``.

    Args:
        env: the ``OvercookedEnv`` the BC model will be deployed in (a ``*_9`` layout,
            ``train_layout_mode='fixed'``). Its reset observation supplies the canonical
            static layout channels.
        native_layout_name: the dataset's native layout name (from
            ``dataset.metadata["layout"]``, e.g. ``"counter_circuit"``). Selects the
            embedding offset; unknown names fall back to identity top-left placement.
        rng: PRNG key used for the one-shot env reset that reads the static layout.
    """

    def __init__(self, env, native_layout_name: str, rng: jax.Array | None = None) -> None:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        # jaxmarl reset -> ({'agent_0': [9,9,26], 'agent_1': ...}, state). Static channels
        # (10/11/12/14/15) are identical for both agents and constant over an episode.
        reset_obs, _ = env._env.reset(rng)
        self._env_static: jax.Array = jnp.asarray(reset_obs["agent_0"], dtype=jnp.float32)
        if self._env_static.ndim != 3:
            raise ValueError(
                f"Expected env obs [H,W,C]; got shape {self._env_static.shape}."
            )
        self.target_h, self.target_w, self.n_channels = self._env_static.shape

        self.native_layout_name: str = native_layout_name
        self._placement: tuple[int, int, int, int, int, int] | None = (
            _NATIVE_TO_9_PLACEMENT.get(native_layout_name)
        )

    def _resolve_placement(
        self, native_h: int, native_w: int
    ) -> tuple[int, int, int, int, int, int]:
        if self._placement is not None:
            return self._placement
        # Unknown layout: identity top-left using the demo's own native size.
        return (0, 0, int(native_h), int(native_w), 0, 0)

    def __call__(self, grid_2d: jax.Array) -> jax.Array:
        """Map ``grid_2d`` ``[..., h, w, 26]`` -> ``[..., H, W, 26]`` (env frame).

        Leading batch dims are preserved.
        """
        grid = jnp.asarray(grid_2d, dtype=jnp.float32)
        native_h, native_w, n_c = grid.shape[-3:]
        if n_c != self.n_channels:
            raise ValueError(
                f"Channel mismatch: demo has {n_c}, env has {self.n_channels}."
            )
        src_r0, src_c0, nr, nc, dst_r0, dst_c0 = self._resolve_placement(native_h, native_w)
        if src_r0 + nr > native_h or src_c0 + nc > native_w:
            raise ValueError(
                f"Placement {self._placement} for layout {self.native_layout_name!r} "
                f"exceeds demo obs size ({native_h}, {native_w})."
            )

        batch_shape = grid.shape[:-3]
        out = jnp.zeros(
            (*batch_shape, self.target_h, self.target_w, self.n_channels), dtype=jnp.float32
        )

        # 1. Place demo (dynamic) content into the embedding region.
        out = out.at[..., dst_r0:dst_r0 + nr, dst_c0:dst_c0 + nc, :].set(
            grid[..., src_r0:src_r0 + nr, src_c0:src_c0 + nc, :]
        )

        # 2. Overwrite static layout channels from the target _9 env (broadcasts over batch).
        static_idx = jnp.asarray(_STATIC_CHANNELS)
        out = out.at[..., static_idx].set(self._env_static[..., static_idx])

        # 3. Broadcast the uniform urgency flag across the full grid. The demo value is
        #    constant over its native grid; read it from the embedded region's origin.
        urgency = grid[..., src_r0, src_c0, _URGENCY_CHANNEL]  # [...]
        out = out.at[..., _URGENCY_CHANNEL].set(urgency[..., None, None])

        return out
