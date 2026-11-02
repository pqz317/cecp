"""Overcooked-specific evaluation helpers: the five original layouts and their fixed start states,
observation hashing, layout descriptions and rendering of single states."""
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.overcooked import COLOR_TO_INDEX, DIR_TO_VEC, OBJECT_TO_INDEX, OvercookedState, overcooked_layouts
from ..envs.spaces import Observation
from .display import OC_ORIGINAL_LAYOUT_NAMES

# The five original Overcooked layouts (9x9 versions), in sorted order: per-layout RNG splits follow it.
ORIGINAL_LAYOUTS: tuple[str, ...] = tuple(sorted(OC_ORIGINAL_LAYOUT_NAMES))
GRID_SIZE = 9           # the original layouts are 9x9
AGENT_VIEW_SIZE = 5     # what create_frame_renderer's padding is parameterised by

# Observation channels holding the *other* agent's orientation (one-hot over 4 directions, set at its
# position). Each agent sees its own layers first (see the vendored Overcooked.get_obs), so this is the
# partner's orientation from either seat.
PARTNER_ORIENTATION_CHANNELS = slice(6, 10)
# Seat-independent pot channels (get_obs docstring).
CH_POT_LOC, CH_ONIONS_IN_POT, CH_POT_COOK_TIME, CH_SOUP_READY = 10, 16, 20, 21

# Index i of the policy's action vector. The vendored env's Actions enum names (right, down, left, up)
# do not match DIR_TO_VEC = [north, south, east, west], and agent_dir_idx = action for move actions,
# so action 0 moves the agent up; `assert_action_semantics` re-derives this by stepping the env.
ACTION_NAMES = ['up', 'down', 'right', 'left', 'stay', 'interact']
ACTION_MOVE_DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]     # (d_row, d_col) of the move actions
ORIENTATION_NAMES = ['up', 'down', 'right', 'left']        # agent_dir_idx shares the move-action indexing
INVENTORY_ITEMS = {'empty': OBJECT_TO_INDEX['empty'], 'onion': OBJECT_TO_INDEX['onion'],
                   'plate': OBJECT_TO_INDEX['plate'], 'dish': OBJECT_TO_INDEX['dish']}
INVENTORY_NAMES = {v: k for k, v in INVENTORY_ITEMS.items()}
# The visualiser rotates the agent triangle assuming (right, down, left, up) tile directions, not
# DIR_TO_VEC's order; the tile value that DRAWS each facing direction:
RENDER_DIR_INDEX = {'right': 0, 'down': 1, 'left': 2, 'up': 3}


# -- fixed start states --------------------------------------------------------------------------

def fixed_initial_state(env: Any, layout_name: str) -> tuple[OvercookedState, list[Observation]]:
    """The deterministic start state of `layout_name` (no random agent placement, no shuffled pots or
    inventories), as (env state, per-seat observations) for a single unbatched `OvercookedEnv`."""
    _, raw_state = env._env.custom_reset(
        jax.random.PRNGKey(0), layout=overcooked_layouts[layout_name], random_reset=False, shuffle_inv_and_pot=False)
    state = OvercookedState(_state=raw_state, _step=jnp.int32(0))
    raw_obs = env._env.get_obs(raw_state)
    obs = Observation({'grid_2d': jnp.stack([raw_obs['agent_0'], raw_obs['agent_1']])})
    if env.pad_obs_shape_to is not None:
        def pad(x: jax.Array) -> jax.Array:
            widths = [(0, 0)] * (x.ndim - 3) + [(0, p - d) for p, d in zip(env.pad_obs_shape_to, x.shape[-3:])]
            return jnp.pad(x, widths, mode='constant', constant_values=0)
        obs = jax.tree_util.tree_map(pad, obs)
    return state, [obs[0:1], obs[1:2]]


def tile_initial_state(state: OvercookedState, obs_lst: list[Observation], n_envs: int) -> tuple[OvercookedState, list[Observation]]:
    """Broadcast a single fixed start state to a batched env's leading env axis."""
    state = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n_envs, *jnp.asarray(x).shape)), state)
    obs_lst = [jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_envs, *x.shape[1:])), obs) for obs in obs_lst]
    return state, obs_lst


def check_tiled_state(state: OvercookedState, obs_lst: list[Observation], batched_env: Any, env_const: Any) -> None:
    """Assert a tiled start state has exactly the structure, shapes and dtypes `batched_env.reset` returns
    (checked abstractly with jax.eval_shape, so nothing is compiled or run)."""
    ref_state, ref_obs = jax.eval_shape(batched_env.reset, jax.random.key(0), env_const)
    if jax.tree_util.tree_structure(state) != jax.tree_util.tree_structure(ref_state):
        raise AssertionError('Tiled start state pytree does not match env.reset.')
    for a, b in zip(jax.tree_util.tree_leaves((state, obs_lst)), jax.tree_util.tree_leaves((ref_state, ref_obs))):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f'Tiled start state leaf is {a.shape}/{a.dtype}, env.reset gives {b.shape}/{b.dtype}.')


# -- observation hashing -------------------------------------------------------------------------

# Three odd multipliers for a polynomial observation hash: 96 bits over a few million rows.
HASH_BASES = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D)


def hash_multipliers(obs_dim: int) -> jax.Array:
    """[3, D] uint32 coefficients base_k ** d mod 2**32."""
    mods = np.empty((len(HASH_BASES), obs_dim), dtype=np.uint32)
    for k, base in enumerate(HASH_BASES):
        value = 1
        for d in range(obs_dim):
            mods[k, d] = np.uint32(value)
            value = (value * base) & 0xFFFFFFFF
    return jnp.asarray(mods)


def hash_obs(obs_flat: jax.Array, multipliers: jax.Array) -> jax.Array:
    """[B, D] integral observations -> [B, 3] uint32 (explicit uint32 accumulation, so it wraps)."""
    x = obs_flat.astype(jnp.uint32)
    return jnp.stack([jnp.sum(x * multipliers[k][None, :], axis=-1, dtype=jnp.uint32) for k in range(multipliers.shape[0])], axis=-1)


def hash_obs_numpy(obs_flat: np.ndarray) -> np.ndarray:
    """Host-side version of `hash_obs` for one [D] observation, independent of the JAX path."""
    out = np.empty(len(HASH_BASES), dtype=np.uint32)
    for k, base in enumerate(HASH_BASES):
        acc, mult = 0, 1
        for v in obs_flat.astype(np.uint64).tolist():
            acc = (acc + v * mult) & 0xFFFFFFFF
            mult = (mult * base) & 0xFFFFFFFF
        out[k] = acc
    return out


def structured_view(arr: np.ndarray) -> np.ndarray:
    """[N, K] -> [N] structured view, so np.unique works on whole rows."""
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype([(f'f{i}', arr.dtype) for i in range(arr.shape[1])])).ravel()


# -- layout descriptions -------------------------------------------------------------------------

def layout_grid(layout_name: str) -> np.ndarray:
    """Character grid of a layout (X counter, G serving, O onion pile, P plate pile, T pot, A agent start)."""
    layout = overcooked_layouts[layout_name]
    height, width = int(layout['height']), int(layout['width'])
    grid = np.full((height, width), '.', dtype='<U1')
    for key, char in [('wall_idx', 'X'), ('goal_idx', 'G'), ('onion_pile_idx', 'O'),
                      ('plate_pile_idx', 'P'), ('pot_idx', 'T'), ('agent_idx', 'A')]:
        for flat in np.asarray(layout.get(key, [])).ravel():
            grid[int(flat) // width, int(flat) % width] = char
    return grid


def pot_cells(layout_name: str) -> list[tuple[int, int]]:
    layout = overcooked_layouts[layout_name]
    width = int(layout['width'])
    return sorted((int(f) // width, int(f) % width) for f in np.asarray(layout['pot_idx']).ravel())


def describe_cell(layout_name: str, row: int, col: int) -> str:
    """E.g. 'pot up at (0, 4)': the interactive tiles next to a cell."""
    grid = layout_grid(layout_name)
    names = {'T': 'pot', 'O': 'onion pile', 'P': 'plate pile', 'G': 'serving counter', 'X': 'counter'}
    neighbours = []
    for (d_row, d_col), facing in zip(ACTION_MOVE_DELTAS, ORIENTATION_NAMES):
        r, c = row + d_row, col + d_col
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] in names:
            neighbours.append(f'{names[grid[r, c]]} {facing} at ({r}, {c})')
    return '; '.join(neighbours) if neighbours else 'open floor'


def assert_action_semantics(env: Any, layout_name: str) -> None:
    """Re-derive action index -> direction by stepping the env from the layout's start state, so
    ACTION_NAMES cannot silently drift from the env."""
    raw_env = env._env
    _, state = raw_env.custom_reset(jax.random.PRNGKey(0), layout=overcooked_layouts[layout_name],
                                    random_reset=False, shuffle_inv_and_pot=False)
    pos0 = np.asarray(state.agent_pos, dtype=np.int64)[0][::-1]   # (x, y) -> (row, col)
    for action, expected in enumerate(ACTION_MOVE_DELTAS):
        _, next_state, _, _, _ = raw_env.step_env(jax.random.PRNGKey(1), state, {'agent_0': jnp.array(action), 'agent_1': jnp.array(4)})
        moved = tuple(np.asarray(next_state.agent_pos, dtype=np.int64)[0][::-1] - pos0)
        if int(np.asarray(next_state.agent_dir_idx)[0]) != action:
            raise AssertionError(f'Action {action} did not set agent_dir_idx={action}; ORIENTATION_NAMES is wrong.')
        if moved != (0, 0) and moved != expected:
            raise AssertionError(f'Action {action} ({ACTION_NAMES[action]}) moved the agent by {moved}, expected {expected}.')


# -- rendering -----------------------------------------------------------------------------------

def render_grid_frame(env: Any, state: OvercookedState) -> np.ndarray:
    """RGB frame of a single (unbatched) state, cropped to exactly the GRID_SIZE x GRID_SIZE grid, so
    it lines up with a heatmap drawn on extent (-0.5, GRID_SIZE - 0.5)."""
    frame = np.asarray(env.create_frame_renderer()(state, AGENT_VIEW_SIZE))
    rendered_tiles = int(np.asarray(state._state.maze_map).shape[0]) - 2 * (AGENT_VIEW_SIZE - 2)
    tile_pixels, remainder = divmod(frame.shape[0], rendered_tiles)
    grid_h, grid_w = int(env._env.obs_shape[1]), int(env._env.obs_shape[0])
    if remainder != 0 or rendered_tiles < max(grid_h, grid_w):
        raise ValueError(f'rendered frame {frame.shape[:2]} is not {rendered_tiles} whole tiles of a {grid_h}x{grid_w} grid')
    pad_rows, pad_cols = (rendered_tiles - grid_h) // 2, (rendered_tiles - grid_w) // 2
    return frame[pad_rows * tile_pixels:(pad_rows + grid_h) * tile_pixels, pad_cols * tile_pixels:(pad_cols + grid_w) * tile_pixels]


def state_for_rendering(state: OvercookedState, red_seat: int = 0) -> OvercookedState:
    """Fix the two agent tiles so the visualiser draws them right: tile direction via RENDER_DIR_INDEX,
    and seat `red_seat` coloured red (the visualiser colours by tile, and looks inventory and direction
    up per colour, so those are swapped along with it)."""
    raw = state._state
    maze_map = np.array(raw.maze_map)
    padding = (maze_map.shape[0] - GRID_SIZE) // 2
    colors = {red_seat: COLOR_TO_INDEX['red'], 1 - red_seat: COLOR_TO_INDEX['blue']}
    for seat in (0, 1):
        x, y = int(raw.agent_pos[seat, 0]), int(raw.agent_pos[seat, 1])
        if int(maze_map[y + padding, x + padding, 0]) != OBJECT_TO_INDEX['agent']:
            raise AssertionError(f'No agent tile at seat {seat} position ({y}, {x}).')
        maze_map[y + padding, x + padding, 1] = colors[seat]
        maze_map[y + padding, x + padding, 2] = RENDER_DIR_INDEX[ORIENTATION_NAMES[int(raw.agent_dir_idx[seat])]]
    updates = {'maze_map': jnp.asarray(maze_map)}
    if red_seat == 1:
        updates['agent_inv'] = jnp.asarray(np.asarray(raw.agent_inv)[::-1].copy())
        updates['agent_dir_idx'] = jnp.asarray(np.asarray(raw.agent_dir_idx)[::-1].copy())
    return state.replace(_state=raw.replace(**updates))


def state_with_single_agent_at(state: OvercookedState, cell: tuple[int, int], orientation: int, inventory: int) -> OvercookedState:
    """`state` with seat 0 (drawn red) moved onto `cell` (row, col), facing `orientation` and holding
    `inventory`, and the other agent removed from the picture. For rendering only, never stepped."""
    raw = state._state
    maze_map = np.asarray(raw.maze_map).copy()
    padding = (maze_map.shape[0] - int(np.asarray(raw.wall_map).shape[0])) // 2
    agent_pos, agent_dir_idx, agent_inv = (np.asarray(x).copy() for x in (raw.agent_pos, raw.agent_dir_idx, raw.agent_inv))
    for seat in range(agent_pos.shape[0]):
        maze_map[padding + int(agent_pos[seat, 1]), padding + int(agent_pos[seat, 0]), :] = [OBJECT_TO_INDEX['empty'], 0, 0]
    row, col = int(cell[0]), int(cell[1])
    maze_map[padding + row, padding + col, :] = [OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'],
                                                 RENDER_DIR_INDEX[ORIENTATION_NAMES[orientation]]]
    agent_pos[0], agent_dir_idx[0], agent_inv[0] = (col, row), orientation, inventory
    return state.replace(_state=raw.replace(
        maze_map=jnp.asarray(maze_map), agent_pos=jnp.asarray(agent_pos, dtype=raw.agent_pos.dtype),
        agent_dir_idx=jnp.asarray(agent_dir_idx, dtype=raw.agent_dir_idx.dtype),
        agent_dir=jnp.asarray(np.asarray(DIR_TO_VEC)[agent_dir_idx], dtype=raw.agent_dir.dtype),
        agent_inv=jnp.asarray(agent_inv, dtype=raw.agent_inv.dtype)))
