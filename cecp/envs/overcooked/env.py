from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from ._jaxmarl.overcooked import Overcooked as _Overcooked, State as _OvercookedState
import numpy as np

from ..base import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace


_LAYOUT_PROCGEN_NAMES = ('cramped_room', 'coord_ring', 'counter_circuit', 'forced_coord', 'asymm_advantages')


def _procgen_generator_map() -> dict[str, Callable[..., Any]]:
    """Maps a base layout name to the jaxmarl 9x9 proc-gen generator for its family."""
    from ._jaxmarl.layouts import (
        make_cramped_room_9x9, make_coord_ring_9x9, make_counter_circuit_9x9,
        make_forced_coord_9x9, make_asymm_advantages_9x9,
    )
    return {
        'cramped_room': make_cramped_room_9x9,
        'coord_ring': make_coord_ring_9x9,
        'counter_circuit': make_counter_circuit_9x9,
        'forced_coord': make_forced_coord_9x9,
        'asymm_advantages': make_asymm_advantages_9x9,
    }


class _SingleFamilyProcgenOvercooked(_Overcooked):
    """Overcooked whose every reset draws a fresh proc-gen layout from one family.

    jaxmarl's `reset` either replays a fixed layout or samples uniformly over all five
    9x9 generators; there is no option for "sample only from generator X". Overriding
    `reset` covers both explicit resets and the auto-reset that `MultiAgentEnv.step`
    performs on episode end, so training sees a new layout from this family every episode.
    """

    def __init__(self, *args, layout_generator: Callable[..., Any], **kwargs):
        super().__init__(*args, **kwargs)
        self._layout_generator = layout_generator

    def reset(self, key: jax.Array, params: Any = None):
        key, rng_layout = jax.random.split(key)
        sampled_layout = self._layout_generator(rng_layout, ik=True)
        obs, state = self.custom_reset(
            key,
            layout=sampled_layout,
            random_reset=False,
            shuffle_inv_and_pot=self.shuffle_inv_and_pot,
        )
        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state)


class OvercookedConst(BaseEnvConst):
    shaped_reward_factor: float = 0.0
    use_reserved_layouts: bool = False

class OvercookedState(BaseEnvState):
    _state: _OvercookedState

class OvercookedEnv(BaseEnv[OvercookedConst, OvercookedState]):
    def __init__(
        self,
        layout: str = 'cramped_room',
        max_steps: int = 400,
        pad_obs_shape_to: Sequence[int] | None = None,
        heldout_layout_seed: int = 0,
        heldout_layout_num_samples: int = 100,
        train_layout_mode: str = 'random',
        train_procgen_seed: int = 0,
    ):
        """
        Args:
            layout: Layout name from jaxmarl's overcooked_layouts dict. Role depends on
                train_layout_mode (see below).
            max_steps: Episode length.
            pad_obs_shape_to: If set, pad observations to this spatial shape (H, W, C).
                Useful when mixing layout sizes.
            heldout_layout_seed: RNG seed for sampling the proc-gen heldout pool
                (only used in 'random' mode).
            heldout_layout_num_samples: Number of proc-gen layouts to include in the
                heldout eval pool (only used in 'random' mode; set 0 to disable).
            train_layout_mode: Controls which layouts are seen during training and eval.

                'random' (default) — each reset draws a new layout uniformly from the
                    five 9x9 proc-gen generators (asymm_advantages, coord_ring,
                    counter_circuit, forced_coord, cramped_room). The `layout` parameter
                    is ignored during training. Eval uses a separate heldout pool of
                    `heldout_layout_num_samples` proc-gen samples. The handcrafted *_9
                    layouts are excluded from that pool (they are evaluated separately),
                    but training still avoids them.

                'fixed' — every reset (train and eval) uses the exact layout named by
                    `layout` (e.g. 'cramped_room_9'). No heldout pool is built; set
                    eval_on_reserved_layouts=False. Agent start positions are taken
                    from the layout dict if present, otherwise randomized per episode.

                'random_family' — like 'random', but every reset draws from a single
                    proc-gen family: the generator named by `layout` (e.g.
                    'asymm_advantages' → make_asymm_advantages_9x9). A new layout is
                    sampled every episode, so training still sees the proc-gen
                    distribution, just restricted to one base layout. Eval uses a
                    heldout pool of `heldout_layout_num_samples` samples from that same
                    generator (seeded by `heldout_layout_seed`); set
                    eval_on_reserved_layouts=True. Supported layout names are the same
                    as for 'fixed_procgen'.

                'fixed_procgen' — one layout is sampled once at init time from the
                    proc-gen generator corresponding to `layout` (e.g. 'cramped_room'
                    → make_cramped_room_9x9) using `train_procgen_seed`. That single
                    layout is used for every training reset. Eval uses a heldout pool
                    containing only the fixed 9x9 layout named `layout + '_9'`
                    (e.g. 'cramped_room_9'); set eval_on_reserved_layouts=True.
                    Supported layout names: cramped_room, coord_ring, counter_circuit,
                    forced_coord, asymm_advantages.

            train_procgen_seed: RNG seed used to sample the single training layout in
                'fixed_procgen' mode. Different seeds yield different layouts. Unused in
                'random_family' mode, where layouts are drawn from the rollout RNG.
        """
        from flax.core.frozen_dict import FrozenDict
        from ._jaxmarl.layouts import overcooked_layouts as layouts

        if train_layout_mode not in ('random', 'fixed', 'fixed_procgen', 'random_family'):
            raise ValueError(
                "train_layout_mode must be 'random', 'fixed', 'fixed_procgen', or "
                f"'random_family', got {train_layout_mode!r}"
            )

        self._layout_name = layout
        self.train_layout_mode = train_layout_mode
        self.train_procgen_seed = train_procgen_seed

        self._layout_generator: Callable[..., Any] | None = None
        if train_layout_mode in ('fixed_procgen', 'random_family'):
            _procgen_map = _procgen_generator_map()
            if layout not in _procgen_map:
                raise ValueError(
                    f"train_layout_mode={train_layout_mode!r} requires a proc-gen-capable layout name, "
                    f"got {layout!r}. Supported: {list(_procgen_map.keys())}"
                )
            self._layout_generator = _procgen_map[layout]

        if train_layout_mode == 'fixed_procgen':
            procgen_layout = self._layout_generator(jax.random.PRNGKey(train_procgen_seed), ik=True)
            self.layout_dict = FrozenDict(procgen_layout)
            self._env: _Overcooked = _Overcooked(layout=self.layout_dict, max_steps=max_steps, random_reset=False)
        elif train_layout_mode == 'random_family':
            # Nominal layout only: every reset resamples from the family generator.
            self.layout_dict = FrozenDict(layouts[layout + '_9'])
            self._env: _Overcooked = _SingleFamilyProcgenOvercooked(
                layout=self.layout_dict,
                max_steps=max_steps,
                random_reset=False,
                layout_generator=self._layout_generator,
            )
        else:
            self.layout_dict = FrozenDict(layouts[layout])
            self._env: _Overcooked = _Overcooked(
                layout=self.layout_dict,
                max_steps=max_steps,
                random_reset=(train_layout_mode == 'random'),
            )

        self.max_steps: int = max_steps
        if heldout_layout_num_samples < 0:
            raise ValueError(f"heldout_layout_num_samples must be >= 0, got {heldout_layout_num_samples}")

        self.heldout_layout_seed = heldout_layout_seed
        self.heldout_layout_num_samples = heldout_layout_num_samples

        self._initialize_layout_state_pools()

        self.pad_obs_shape_to = tuple(pad_obs_shape_to) if pad_obs_shape_to is not None else None

    def _stack_state_list(self, state_list: list[_OvercookedState]) -> _OvercookedState:
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *state_list)

    def _sample_state_from_pool(self, rng: jax.Array, state_pool: _OvercookedState) -> _OvercookedState:
        pool_size = state_pool.time.shape[0]
        idx = jax.random.randint(rng, shape=(), minval=0, maxval=pool_size)
        return jax.tree_util.tree_map(lambda x: x[idx], state_pool)

    def _generate_heldout_layout_states(
        self,
        layout_generators: list[Callable[..., Any]] | None = None,
    ) -> list[_OvercookedState]:
        if layout_generators is None:
            try:
                generator_map = _procgen_generator_map()
            except Exception:
                return []
            layout_generators = [
                generator_map['asymm_advantages'],
                generator_map['coord_ring'],
                generator_map['counter_circuit'],
                generator_map['forced_coord'],
                generator_map['cramped_room'],
            ]

        if self.heldout_layout_num_samples == 0:
            return []
        key = jax.random.PRNGKey(self.heldout_layout_seed)
        generated_states: list[_OvercookedState] = []
        for _ in range(self.heldout_layout_num_samples):
            key, rng_idx = jax.random.split(key)
            gen_idx = int(jax.random.randint(rng_idx, shape=(), minval=0, maxval=len(layout_generators)))
            key, rng_layout = jax.random.split(key)
            sampled_layout = layout_generators[gen_idx](rng_layout, ik=True)
            key, rng_reset = jax.random.split(key)
            _, sampled_state = self._env.custom_reset(
                rng_reset,
                layout=sampled_layout,
                random_reset=False,
                shuffle_inv_and_pot=False,
            )
            generated_states.append(sampled_state)
        return generated_states

    def _build_handcrafted_layout_states(self) -> list[_OvercookedState]:
        handcrafted_states: list[_OvercookedState] = []
        from ._jaxmarl.layouts import overcooked_layouts as layouts
        for layout_name, layout_dict in layouts.items():
            if "9" not in layout_name:
                continue
            _, sampled_state = self._env.custom_reset(
                jax.random.PRNGKey(0),
                layout=layout_dict,
                random_reset=False,
                shuffle_inv_and_pot=False,
            )
            handcrafted_states.append(sampled_state)
        return handcrafted_states

    def _initialize_layout_state_pools(self) -> None:
        def _set_empty_heldout_buffers() -> None:
            self._env.held_out_goal = jnp.zeros((0, 2), dtype=jnp.uint32)
            self._env.held_out_pot = jnp.zeros((0, 2), dtype=jnp.uint32)
            self._env.held_out_wall = jnp.zeros((0, self._env.height, self._env.width), dtype=jnp.bool_)

        if self.train_layout_mode == 'fixed':
            # No reserved pool — eval uses the same fixed layout via default reset.
            self._heldout_layout_state_pool = None
            self.num_heldout_layouts = 0
            _set_empty_heldout_buffers()
            return

        if self.train_layout_mode == 'random_family':
            # Eval pool: proc-gen samples from the same family the agent trains on.
            heldout_gen_states = self._generate_heldout_layout_states(
                layout_generators=[self._layout_generator]
            )
            self._heldout_layout_state_pool = (
                self._stack_state_list(heldout_gen_states) if len(heldout_gen_states) > 0 else None
            )
            self.num_heldout_layouts = len(heldout_gen_states)

            rejection_states = self._build_handcrafted_layout_states() + heldout_gen_states
            rejection_pool = self._stack_state_list(rejection_states)
            self._env.held_out_goal = rejection_pool.goal_pos
            self._env.held_out_wall = rejection_pool.wall_map
            self._env.held_out_pot = rejection_pool.pot_pos
            return

        if self.train_layout_mode == 'fixed_procgen':
            # Eval pool: the single fixed 9x9 version of the specified layout.
            from ._jaxmarl.layouts import overcooked_layouts as layouts
            eval_layout_name = self._layout_name + '_9'
            if eval_layout_name not in layouts:
                raise ValueError(
                    f"train_layout_mode='fixed_procgen' expects '{eval_layout_name}' in overcooked_layouts "
                    f"as the eval target, but it was not found."
                )
            _, eval_state = self._env.custom_reset(
                jax.random.PRNGKey(0),
                layout=layouts[eval_layout_name],
                random_reset=False,
                shuffle_inv_and_pot=False,
            )
            self._heldout_layout_state_pool = self._stack_state_list([eval_state])
            self.num_heldout_layouts = 1
            self._env.held_out_goal = self._heldout_layout_state_pool.goal_pos
            self._env.held_out_wall = self._heldout_layout_state_pool.wall_map
            self._env.held_out_pot = self._heldout_layout_state_pool.pot_pos
            return

        # random mode: eval samples only from the proc-gen heldout pool. The handcrafted
        # *_9 layouts are evaluated separately (evaluate_cross_play's original-layout
        # section), so they are kept out of the sampling pool. They stay in the train-time
        # rejection buffers below, so training still never resets onto them.
        heldout_gen_states = self._generate_heldout_layout_states()
        self._heldout_layout_state_pool = self._stack_state_list(heldout_gen_states) if len(heldout_gen_states) > 0 else None
        self.num_heldout_layouts = len(heldout_gen_states)

        rejection_states = self._build_handcrafted_layout_states() + heldout_gen_states
        if len(rejection_states) > 0:
            rejection_pool = self._stack_state_list(rejection_states)
            self._env.held_out_goal = rejection_pool.goal_pos
            self._env.held_out_wall = rejection_pool.wall_map
            self._env.held_out_pot = rejection_pool.pot_pos
        else:
            _set_empty_heldout_buffers()
    
    @property
    def default_const(self) -> OvercookedConst:
        return OvercookedConst(
            max_steps=self._env.max_steps,
            use_reserved_layouts=False,
        )

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        if self.pad_obs_shape_to is None:
            obs_space = (self._env.height, self._env.width, 26)
        else:
            obs_space = self.pad_obs_shape_to
        return ObservationSpace({'grid_2d': obs_space}), {'grid_2d': (2,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(self._env.action_set)
        return ActionSpace({'all': num_actions}), {'all': (2,)}
    
    def env_reset(
        self,
        rng: jax.Array,
        const: OvercookedConst
    ) -> tuple[OvercookedState, Observation]:
        if self._heldout_layout_state_pool is None:
            obs, _state = self._env.reset(rng)
        else:
            use_reserved_layouts = jnp.asarray(const.use_reserved_layouts, dtype=jnp.bool_)

            def _reset_from_reserved(_: None) -> tuple[dict[str, jax.Array], _OvercookedState]:
                _state = self._sample_state_from_pool(rng, self._heldout_layout_state_pool)
                obs = self._env.get_obs(_state)
                return obs, _state

            def _reset_from_default(_: None) -> tuple[dict[str, jax.Array], _OvercookedState]:
                return self._env.reset(rng)

            obs, _state = jax.lax.cond(use_reserved_layouts, _reset_from_reserved, _reset_from_default, operand=None)

        state = OvercookedState(_state=_state, _step=0)
        grid_2d = jnp.stack([obs['agent_0'], obs['agent_1']])
        obs = Observation({'grid_2d': grid_2d})

        if self.pad_obs_shape_to is not None:
            def _padding(x: jax.Array) -> jax.Array:
                return jnp.pad(
                    x,
                    pad_width =[(0, 0) for _ in range(x.ndim - 3)] + [(0, pad_d - x_d) for pad_d, x_d in zip(self.pad_obs_shape_to, x.shape[-3:])],
                    mode='constant', constant_values=0
                )
            obs = jax.tree_util.tree_map(_padding, obs)

        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)
    
    def env_step(
        self,
        rng: jax.Array,
        const: OvercookedConst,
        state: OvercookedState,
        action: Action
    ) -> tuple[OvercookedState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        # Return: [state, obs, reward, done, info]
        # For all of obs, reward, done, the shape must follows [num_agents, *]
        env_action = {
            'agent_0': action['all'][0],
            'agent_1': action['all'][1]
        }
        obs, states, rewards, dones, infos = self._env.step(
            key=rng,
            state=state._state,
            actions=env_action,
        ) # We use self._env.step so it automatically applies reset when done
        grid_2d = jnp.stack([obs['agent_0'], obs['agent_1']])
        obs = Observation({'grid_2d': grid_2d})
        state = OvercookedState(_state=states, _step=state._step+1)
        reward = jnp.stack([rewards['agent_0'], rewards['agent_1']])
        shaped_reward = infos['shaped_reward']
        shaped_reward = jnp.stack([shaped_reward['agent_0'], shaped_reward['agent_1']])
        reward += const.shaped_reward_factor * shaped_reward
        done = jnp.stack([dones['agent_0'], dones['agent_1']])
        info = {'_overcooked_env_info': infos}

        if self.pad_obs_shape_to is not None:
            def _padding(x: jax.Array) -> jax.Array:
                return jnp.pad(
                    x,
                    pad_width =[(0, 0) for _ in range(x.ndim - 3)] + [(0, pad_d - x_d) for pad_d, x_d in zip(self.pad_obs_shape_to, x.shape[-3:])],
                    mode='constant', constant_values=0
                )
            obs = jax.tree_util.tree_map(_padding, obs)

        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs), jax.lax.stop_gradient(reward), jax.lax.stop_gradient(done), info
    
    @property
    def num_agents(self) -> int:
        return 2
    
    def get_agent_batch_size(self) -> list[int]:
        return [1 for _ in range(self.num_agents)]

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self.env_observation_space[0] for _ in range(2)]
    
    def get_action_space(self) -> list[ActionSpace]:
        return [self.env_action_space[0] for _ in range(2)]

    def reset(
        self,
        rng: jax.Array,
        const: OvercookedConst
    ) -> tuple[OvercookedState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[0:1], env_obs[1:2]]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: OvercookedConst,
        state: OvercookedState,
        action: Sequence[Action]
    ) -> tuple[OvercookedState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.concatenate([action[0]['all'], action[1]['all']])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[0:1], env_obs[1:2]]
        reward_lst = [env_reward[0:1], env_reward[1:2]]
        done_lst = [env_done[0:1], env_done[1:2]]
        return state, obs_lst, reward_lst, done_lst, info

    def create_frame_renderer(self) -> Callable[[OvercookedState, int], np.ndarray]:
        """
        Creates a frame rendering function tailored for Overcooked game states.

        Returns:
            Callable[[OvercookedState, int], np.ndarray]: A function that takes a game state and agent view size,
                                                            and returns a rendered frame as a NumPy array.
        """
        from ._jaxmarl.overcooked_visualizer import OvercookedVisualizer
        def render_frame(state: OvercookedState, agent_view_size: int) -> np.ndarray:
            """Renders a single frame from the game state.
            Args:
                state (OvercookedState): The current state of the Overcooked game.
                agent_view_size (int): The size of the agent's view.
            Returns:
                np.ndarray: The rendered frame as a NumPy array.
            """
            TILE_SIZE_PIXELS = 32
            state = state._state
            padding = agent_view_size - 2
            # Extract the relevant portion of the maze map based on agent view size
            maze_grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            # Render the maze grid into a frame
            frame = OvercookedVisualizer._render_grid(
                maze_grid,
                tile_size=TILE_SIZE_PIXELS,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )
            return frame
        return render_frame

    def export_gif(
        self,
        state_sequence: Sequence[OvercookedState],
        filename: str,
        agent_view_size: int = 5,
        fps: int = 5
    ) -> None:
        from PIL import Image
        from tqdm import tqdm

        render_frame = self.create_frame_renderer()

        frames = [render_frame(state, agent_view_size).astype(np.uint8) for state in state_sequence]
        pil_images = [Image.fromarray(f) for f in frames]
        
        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=1000 // fps,
            loop=0,
            optimize=True,
            save_all_plugin="GIF" 
        )
        tqdm.write(f"\nVisualizing: GIF successfully exported to {filename}\n")

    def _render_predicted_obs_frame(self, pred_obs: Observation, tile_size: int = 32) -> np.ndarray:
        from ._jaxmarl.common import COLOR_TO_INDEX, OBJECT_TO_INDEX
        from ._jaxmarl.overcooked_visualizer import OvercookedVisualizer

        grid_2d = np.asarray(pred_obs['grid_2d'])
        if grid_2d.ndim == 4:
            grid_2d = grid_2d[0]
        if grid_2d.ndim != 3:
            raise ValueError(
                f"pred_obs['grid_2d'] is expected to have shape [H, W, C] or [B, H, W, C], got {grid_2d.shape}"
            )

        h, w, c = grid_2d.shape
        target_h, target_w = self._env.height, self._env.width
        if h != target_h or w != target_w:
            if h < target_h or w < target_w:
                raise ValueError(
                    f"pred_obs spatial shape {(h, w)} is smaller than env shape {(target_h, target_w)}"
                )
            grid_2d = grid_2d[:target_h, :target_w, :]

        # Overcooked observation channel semantics (agent-0 perspective)
        # 0/1: agent positions (self/other)
        # 2:6, 6:10: direction one-hot for self/other
        # 10..: env layers (pots, counters, piles, goals, items, pot-status layers, ...)
        if c < 26:
            raise ValueError(f"pred_obs['grid_2d'] expected >=26 channels, got {c}")

        obj_empty = OBJECT_TO_INDEX['empty']
        obj_wall = OBJECT_TO_INDEX['wall']
        obj_goal = OBJECT_TO_INDEX['goal']
        obj_onion_pile = OBJECT_TO_INDEX['onion_pile']
        obj_plate_pile = OBJECT_TO_INDEX['plate_pile']
        obj_pot = OBJECT_TO_INDEX['pot']
        obj_onion = OBJECT_TO_INDEX['onion']
        obj_plate = OBJECT_TO_INDEX['plate']
        obj_dish = OBJECT_TO_INDEX['dish']
        obj_agent = OBJECT_TO_INDEX['agent']

        color_black = COLOR_TO_INDEX['black']
        color_grey = COLOR_TO_INDEX['grey']
        color_green = COLOR_TO_INDEX['green']
        color_yellow = COLOR_TO_INDEX['yellow']
        color_white = COLOR_TO_INDEX['white']
        color_red = COLOR_TO_INDEX['red']
        color_blue = COLOR_TO_INDEX['blue']

        grid = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        grid[..., 0] = obj_empty
        grid[..., 1] = color_black
        grid[..., 2] = 0

        pot_layer = grid_2d[..., 10]
        wall_layer = grid_2d[..., 11]
        onion_pile_layer = grid_2d[..., 12]
        plate_pile_layer = grid_2d[..., 14]
        goal_layer = grid_2d[..., 15]
        onions_in_pot_layer = grid_2d[..., 16]
        onions_in_soup_layer = grid_2d[..., 18]
        pot_time_layer = grid_2d[..., 20]
        soup_ready_layer = grid_2d[..., 21]
        plate_layer = grid_2d[..., 22]
        onion_layer = grid_2d[..., 23]

        wall_mask = wall_layer > 0.5
        goal_mask = goal_layer > 0.5
        onion_pile_mask = onion_pile_layer > 0.5
        plate_pile_mask = plate_pile_layer > 0.5
        pot_mask = pot_layer > 0.5
        plate_mask = plate_layer > 0.5
        onion_mask = onion_layer > 0.5
        soup_mask = (soup_ready_layer > 0.5) & (~pot_mask)

        grid[wall_mask, 0] = obj_wall
        # Walls are drawn with their color channel (see OvercookedVisualizer._render_obj),
        # so match the grey used by make_overcooked_map for the true-state frames.
        grid[wall_mask, 1] = color_grey
        grid[goal_mask, 0] = obj_goal
        grid[goal_mask, 1] = color_green
        grid[onion_pile_mask, 0] = obj_onion_pile
        grid[onion_pile_mask, 1] = color_yellow
        grid[plate_pile_mask, 0] = obj_plate_pile
        grid[plate_pile_mask, 1] = color_white
        grid[pot_mask, 0] = obj_pot
        grid[plate_mask, 0] = obj_plate
        grid[plate_mask, 1] = color_white
        grid[onion_mask, 0] = obj_onion
        grid[onion_mask, 1] = color_yellow
        grid[soup_mask, 0] = obj_dish
        grid[soup_mask, 1] = color_white

        # Pot status channel in maze_map uses 23..0 encoding.
        inferred_pot_status = np.full((target_h, target_w), 23, dtype=np.uint8)
        num_onions = np.clip(np.rint(onions_in_pot_layer), 0, 3).astype(np.int32)
        inferred_pot_status = np.where(pot_mask, (23 - num_onions).astype(np.uint8), inferred_pot_status)
        cooking_time = np.clip(np.rint(pot_time_layer), 0, 19).astype(np.int32)
        inferred_pot_status = np.where((pot_mask) & (cooking_time > 0), cooking_time.astype(np.uint8), inferred_pot_status)
        inferred_pot_status = np.where((pot_mask) & (soup_ready_layer > 0.5), np.uint8(0), inferred_pot_status)
        inferred_pot_status = np.where((pot_mask) & (onions_in_soup_layer > 0.5) & (cooking_time <= 0), np.uint8(0), inferred_pot_status)
        grid[pot_mask, 2] = inferred_pot_status[pot_mask]

        def _argmax_pos(mask: np.ndarray) -> tuple[int, int]:
            idx = int(np.argmax(mask))
            y, x = np.unravel_index(idx, mask.shape)
            return int(y), int(x)

        self_pos_layer = grid_2d[..., 0]
        other_pos_layer = grid_2d[..., 1]
        self_y, self_x = _argmax_pos(self_pos_layer)
        other_y, other_x = _argmax_pos(other_pos_layer)

        self_dir_idx = int(np.argmax(np.sum(grid_2d[..., 2:6], axis=(0, 1))))
        other_dir_idx = int(np.argmax(np.sum(grid_2d[..., 6:10], axis=(0, 1))))
        agent_dir_idx = np.asarray([self_dir_idx, other_dir_idx], dtype=np.uint8)

        # Infer held inventory from item/soup layers at agent positions.
        def _infer_inventory(y: int, x: int) -> int:
            if soup_ready_layer[y, x] > 0.5 and onions_in_soup_layer[y, x] > 0.5:
                return obj_dish
            if plate_layer[y, x] > 0.5:
                return obj_plate
            if onion_layer[y, x] > 0.5:
                return obj_onion
            return obj_empty

        agent_inv = np.asarray([
            _infer_inventory(self_y, self_x),
            _infer_inventory(other_y, other_x),
        ], dtype=np.uint8)

        grid[self_y, self_x, 0] = obj_agent
        grid[self_y, self_x, 1] = color_red
        grid[self_y, self_x, 2] = self_dir_idx

        grid[other_y, other_x, 0] = obj_agent
        grid[other_y, other_x, 1] = color_blue
        grid[other_y, other_x, 2] = other_dir_idx

        return OvercookedVisualizer._render_grid(
            grid,
            tile_size=tile_size,
            highlight_mask=None,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
        )

    def export_predictive_comparison_gif(
        self,
        state_sequence: Sequence[OvercookedState],
        predicted_next_obs_seq: Sequence[Observation],
        filename: str,
        agent_view_size: int = 5,
        fps: int = 5,
    ) -> None:
        from PIL import Image, ImageDraw, ImageFont
        from tqdm import tqdm

        render_frame = self.create_frame_renderer()
        num_frames = min(len(state_sequence), len(predicted_next_obs_seq))
        if num_frames == 0:
            raise ValueError('state_sequence and predicted_next_obs_seq must both be non-empty')

        pad = 16
        frames: list[np.ndarray] = []
        font = ImageFont.load_default()

        for frame_idx in range(num_frames):
            true_frame = render_frame(state_sequence[frame_idx], agent_view_size).astype(np.uint8)
            pred_frame = self._render_predicted_obs_frame(predicted_next_obs_seq[frame_idx], tile_size=32)

            true_img = Image.fromarray(true_frame)
            pred_img = Image.fromarray(pred_frame)

            draw_true = ImageDraw.Draw(true_img)
            draw_pred = ImageDraw.Draw(pred_img)
            draw_true.text((8, 8), 'Environment', fill=(255, 255, 255), font=font)
            draw_pred.text((8, 8), 'Predicted next obs (agent 0)', fill=(255, 255, 255), font=font)

            canvas_w = true_img.width + pad + pred_img.width
            canvas_h = max(true_img.height, pred_img.height)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
            canvas.paste(true_img, (0, 0))
            canvas.paste(pred_img, (true_img.width + pad, 0))

            draw_canvas = ImageDraw.Draw(canvas)
            draw_canvas.text((8, canvas_h - 28), f'Step: {frame_idx + 1} / {num_frames}', fill=(0, 0, 0), font=font)
            frames.append(np.asarray(canvas))

        pil_images = [Image.fromarray(f) for f in frames]
        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=1000 // fps,
            loop=0,
            optimize=True,
            save_all_plugin='GIF'
        )
        tqdm.write(f"\nVisualizing: predictive comparison GIF successfully exported to {filename}\n")
        