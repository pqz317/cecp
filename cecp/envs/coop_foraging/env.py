from enum import IntEnum
from typing import Any, Sequence

from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from ..base import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace
from .visualizer import CoopForagingVisualizer

class CoopForagingConst(BaseEnvConst):
    goal_reward: float
    step_penalty: float
    use_reserved_goal_positions: bool = False

class AgentState(PyTreeNode):
    pos: jax.Array
    last_step_reward: jax.Array

class CoopForagingState(BaseEnvState):
    agent_state: AgentState
    goal_pos: jax.Array
    wall_map: jax.Array

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

class CoopForagingEnv(BaseEnv[CoopForagingConst, CoopForagingState]):
    def __init__(
        self,
        num_agents: int = 2,
        num_goals: int = 2,
        grid_size: int = 10,
        max_steps: int = 50,
        goal_reward: float = 10.0,
        step_penalty: float = -0.1,
        obstacle_density: float = 0.1,
        random_reset: bool = True,
        heldout_goal_ratio: float = 0.0,
        heldout_goal_seed: int = 0,
        train_layout_mode: str | None = None,
        train_procgen_seed: int = 0,
    ) -> None:
        """
        Initializes the cooperative foraging environment.
        Args:
            num_agents (int, optional): Number of agents in the environment. Defaults to 2.
            num_goals (int, optional): Number of goals in the environment. Defaults to 2.
            grid_size (int, optional): Size of the grid (grid is square, so this is both width and height). Defaults to 10.
            max_steps (int, optional): Maximum number of steps before the episode ends. Defaults to 50.
            goal_reward (float, optional): Reward given to agents for reaching a goal. Defaults to 10.0.
            step_penalty (float, optional): Penalty applied for each step taken. Defaults to -0.1.
            obstacle_density (float, optional): Density of obstacles in the grid, as a fraction of available positions. Defaults to 0.1.
            random_reset (bool, optional): Deprecated alias. Use train_layout_mode instead. If True → 'random', False → 'fixed_procgen'. Ignored when train_layout_mode is set.
            heldout_goal_ratio (float, optional): Fraction of available non-wall positions reserved exclusively for test-time goal sampling. Training-time sampling excludes these positions. Defaults to 0.0.
            heldout_goal_seed (int, optional): Seed used once at initialization to pick reserved goal positions. Defaults to 0.
            train_layout_mode (str | None, optional): Controls which layouts are seen during training.
                'random' — each reset draws new goal/obstacle positions (equivalent to random_reset=True).
                'fixed_procgen' — goal/obstacle positions are sampled once at init time using train_procgen_seed and never change across resets.
                None (default) — falls back to random_reset parameter.
            train_procgen_seed (int, optional): RNG seed used to sample the fixed layout in 'fixed_procgen' mode. Defaults to 0.
        Attributes:
            n_agent (int): Number of agents in the environment.
            n_goal (int): Number of goals in the environment.
            grid_size (int): Size of the grid.
            max_steps (int): Maximum number of steps in an episode.
            goal_reward (float): Reward for reaching a goal.
            step_penalty (float): Penalty for each step taken.
            n_obstacle (int): Number of obstacles in the grid, calculated based on obstacle density.
            available_positions (jnp.ndarray): Array of available positions in the grid (excluding borders).
        """
        if train_layout_mode is not None and train_layout_mode not in ('random', 'fixed_procgen'):
            raise ValueError(f"train_layout_mode must be 'random' or 'fixed_procgen', got {train_layout_mode!r}")
        if train_layout_mode is None:
            train_layout_mode = 'random' if random_reset else 'fixed_procgen'
        self.train_layout_mode = train_layout_mode
        self.train_procgen_seed = train_procgen_seed
        self.random_reset = (train_layout_mode == 'random')

        self.n_agent = num_agents
        self.n_goal = num_goals
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.n_obstacle = int(obstacle_density * (grid_size - 2) * (grid_size - 2))
        h, w = self.grid_size, self.grid_size
        self.available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        if len(self.available_positions) < self.n_goal + self.n_agent + self.n_obstacle:
            raise ValueError(
                f"Not enough available positions ({len(self.available_positions)}) for "
                f"n_goal({self.n_goal}) + n_agent({self.n_agent}) + n_obstacle({self.n_obstacle})."
            )
        if heldout_goal_ratio < 0.0 or heldout_goal_ratio > 1.0:
            raise ValueError(f"heldout_goal_ratio must be in [0, 1], got {heldout_goal_ratio}")
        self.heldout_goal_ratio = heldout_goal_ratio
        num_positions = len(self.available_positions)
        self.num_heldout_goal_positions = int(round(num_positions * heldout_goal_ratio))
        if self.num_heldout_goal_positions > 0 and self.num_heldout_goal_positions < self.n_goal:
            raise ValueError(
                f"heldout_goal_ratio={heldout_goal_ratio} yields only {self.num_heldout_goal_positions} held-out positions, "
                f"but n_goal={self.n_goal}. Increase heldout_goal_ratio or reduce n_goal."
            )
        rng_partition = jax.random.key(heldout_goal_seed)
        if self.num_heldout_goal_positions > 0:
            heldout_indices = jax.random.choice(
                rng_partition,
                num_positions,
                shape=(self.num_heldout_goal_positions,),
                replace=False,
            )
            heldout_mask = jnp.zeros((num_positions,), dtype=jnp.bool_).at[heldout_indices].set(True)
        else:
            heldout_mask = jnp.zeros((num_positions,), dtype=jnp.bool_)
        self._heldout_goal_mask = heldout_mask
        self._heldout_goal_positions = self.available_positions[self._heldout_goal_mask]
        self._train_goal_positions = self.available_positions[~self._heldout_goal_mask]
        if len(self._train_goal_positions) < self.n_goal:
            raise ValueError(
                f"Not enough train-goal positions after holdout split: {len(self._train_goal_positions)} < n_goal={self.n_goal}. "
                f"Reduce heldout_goal_ratio or n_goal."
            )

        if train_layout_mode == 'fixed_procgen':
            _, fixed_goal_pos, fixed_obstacle_pos = self._reset_positions(
                jax.random.PRNGKey(train_procgen_seed), use_reserved_goal_positions=False
            )
            self._fixed_goal_pos = fixed_goal_pos
            self._fixed_obstacle_pos = fixed_obstacle_pos
        else:
            self._fixed_goal_pos = None
            self._fixed_obstacle_pos = None
        self.visualizer = CoopForagingVisualizer(self)

    @property
    def default_const(self) -> CoopForagingConst:
        return CoopForagingConst(
            goal_reward=self.goal_reward,
            step_penalty=self.step_penalty,
            use_reserved_goal_positions=False,
            max_steps=self.max_steps
        )

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        """
        Currently, just return space of grid size x grid size x num_channels
        """
        grid_num_channels = {
            'wall': 1,
            'my_pos': 1,
            'goal_pos': self.n_goal,
            'all_agent_pos': self.n_agent,
        }
        grid_obs_space = (self.grid_size, self.grid_size, sum(grid_num_channels.values()))  # Wall and agent positions
        return ObservationSpace({'grid': grid_obs_space, 'agent_info': (1, )}), {'grid': (self.n_agent,), 'agent_info': (self.n_agent,)}

    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(Actions)
        return ActionSpace({'all': num_actions}), {'all': (self.n_agent,)}

    def _reset_positions(self, rng: jax.Array, use_reserved_goal_positions: bool | jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Returns random negative positions for agents, goals, and obstacles. 
        Ensures that all positions are unique and do not overlap with walls 
        (which are at the borders of the grid).
        """
        rng, rng_goal, rng_other = jax.random.split(rng, 3)
        goal_candidate_mask = jnp.where(use_reserved_goal_positions, self._heldout_goal_mask, ~self._heldout_goal_mask)
        goal_probs = goal_candidate_mask.astype(jnp.float32)
        goal_probs = goal_probs / goal_probs.sum()
        goal_indices = jax.random.choice(
            rng_goal,
            len(self.available_positions),
            shape=(self.n_goal,),
            replace=False,
            p=goal_probs,
        )
        goal_pos = self.available_positions[goal_indices]

        is_goal_pos = jnp.any(
            jnp.all(self.available_positions[:, None, :] == goal_pos[None, :, :], axis=-1),
            axis=1,
        )
        non_goal_probs = (~is_goal_pos).astype(jnp.float32)
        non_goal_probs = non_goal_probs / non_goal_probs.sum()
        other_indices = jax.random.choice(
            rng_other,
            len(self.available_positions),
            shape=(self.n_agent + self.n_obstacle,),
            replace=False,
            p=non_goal_probs,
        )
        agent_indices, obstacle_indices = other_indices[:self.n_agent], other_indices[self.n_agent:]
        agent_pos = self.available_positions[agent_indices]
        obstacle_pos = self.available_positions[obstacle_indices]
        return agent_pos, goal_pos, obstacle_pos

    def _reset_agent_positions(self, rng: jax.Array, goal_pos: jax.Array, obstacle_pos: jax.Array) -> jax.Array:
        """
        Resets the positions of the agents, ensuring they do not overlap with goals or obstacles.
        """
        goal_mask = jnp.any(
            jnp.all(self.available_positions[:, None, :] == goal_pos[None, :, :], axis=-1),
            axis=1,
        )
        obstacle_mask = jnp.any(
            jnp.all(self.available_positions[:, None, :] == obstacle_pos[None, :, :], axis=-1),
            axis=1,
        )
        occupied_mask = jnp.logical_or(goal_mask, obstacle_mask)
        valid_probs = (~occupied_mask).astype(jnp.float32)
        valid_probs = valid_probs / valid_probs.sum()
        indices = jax.random.choice(
            rng,
            len(self.available_positions),
            shape=(self.n_agent,),
            replace=False,
            p=valid_probs,
        )
        return self.available_positions[indices]

    def env_reset(self, rng: jax.Array, const: CoopForagingConst) -> tuple[CoopForagingConst, Observation]:
        if self.random_reset:
            rng, rng_reset_pos = jax.random.split(rng)
            agent_pos, goal_pos, obstacle_pos = self._reset_positions(rng_reset_pos, const.use_reserved_goal_positions)
        else:
            if self._fixed_goal_pos is None or self._fixed_obstacle_pos is None:
                raise RuntimeError("fixed_procgen mode requires precomputed fixed goal/obstacle positions")
            goal_pos, obstacle_pos = self._fixed_goal_pos, self._fixed_obstacle_pos

            rng, rng_reset_agent_pos = jax.random.split(rng)
            agent_pos = self._reset_agent_positions(rng_reset_agent_pos, goal_pos, obstacle_pos)

        h, w = self.grid_size, self.grid_size
        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True).at[-1, :].set(True).at[:, 0].set(True).at[:, -1].set(True)
        wall_map = wall_map.at[obstacle_pos[:, 0], obstacle_pos[:, 1]].set(True)

        state = CoopForagingState(
            _step=0,
            agent_state=AgentState(
                pos=agent_pos,
                last_step_reward=jnp.zeros((self.n_agent,), dtype=jnp.float32),
            ),
            goal_pos=goal_pos,
            wall_map=wall_map,
        )
        obs = self.get_obs(state)
        return stop_gradient(state), stop_gradient(obs)
    
    def get_agent_obs(self, state: CoopForagingState, agent_state: AgentState, agent_id: int) -> Observation:
        h, w = self.grid_size, self.grid_size
        agent_pos = agent_state.pos

        grid_channels = {
            'wall': state.wall_map.reshape(h, w, 1).astype(jnp.float32),
            'my_pos': jnp.zeros((h, w, 1), dtype=jnp.float32).at[agent_pos[0], agent_pos[1], 0].set(1),
            'goal_pos': jnp.zeros((h, w, self.n_goal), dtype=jnp.float32).at[state.goal_pos[:, 0], state.goal_pos[:, 1], jnp.arange(self.n_goal)].set(1),
            **({
                'all_agent_pos': jnp.zeros((h, w, self.n_agent), dtype=jnp.float32).at[state.agent_state.pos[:, 0], state.agent_state.pos[:, 1], jnp.arange(self.n_agent)].set(1)
            })
        }
        grid_obs = jnp.concatenate([v for v in grid_channels.values()], axis=-1)
        agent_info_obs = jnp.array([agent_state.last_step_reward])
        return Observation({'grid': grid_obs, 'agent_info': agent_info_obs})

    def get_obs(self, state: CoopForagingState) -> Observation:
        return jax.vmap(self.get_agent_obs, in_axes=(None, 0, 0))(state, state.agent_state, jnp.arange(self.n_agent))

    def agent_step(self, rng: jax.Array, const: CoopForagingConst, state: CoopForagingState, agent_state: AgentState, action: int) -> dict[str, jax.Array]:
        agent_pos = agent_state.pos
        move = jnp.array([0, 0])
        move = jax.lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        new_pos = agent_pos + move
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = jax.lax.cond(collides, lambda _: agent_pos, lambda _: new_pos, None)

        on_goal = lambda x, y: jnp.all(x == y)
        is_goal_hit = jax.vmap(on_goal, in_axes=(None, 0))(new_pos, state.goal_pos)

        return {'new_pos': new_pos, "is_goal_hit": is_goal_hit}

    def env_step(
        self, 
        rng: jax.Array, 
        const: CoopForagingConst, 
        state: CoopForagingState, 
        action: Action
    ) -> tuple[CoopForagingState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = action['all']
        rng, rng_agent_step_batch = jax.random.split(rng)
        rng_agent_step_batch = jax.random.split(rng_agent_step_batch, self.n_agent)
        data = jax.vmap(self.agent_step, in_axes=(0, None, None, 0, 0))(rng_agent_step_batch, const, state, state.agent_state, env_action)
        # print("is goal hit: ")
        # print(data["is_goal_hit"])

        all_agents_same_goal = jnp.any(jnp.logical_and(data["is_goal_hit"][0, :], data["is_goal_hit"][1, :]))
        reward = const.goal_reward * all_agents_same_goal + const.step_penalty
        reward = jnp.full((self.n_agent,), reward, dtype=jnp.float32)
        # print(reward)
        new_state = state.replace(
            _step=state._step + 1,
            agent_state=AgentState(
                pos=data['new_pos'],
                last_step_reward=reward,
            )
        )

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)
        num_steps = state._step + 1
        done = num_steps >= const.max_steps
        s, o, r, d, info = jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {}),
            None
        )
        return stop_gradient(s), stop_gradient(o), stop_gradient(r), stop_gradient(d), stop_gradient(info)
    
    @property
    def num_agents(self) -> int:
        return self.n_agent

    def get_agent_batch_size(self) -> list[int]:
        return [1 for _ in range(self.num_agents)]

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self.env_observation_space[0] for _ in range(self.n_agent)]
    
    def get_action_space(self) -> list[ActionSpace]:
        return [self.env_action_space[0] for _ in range(self.n_agent)]

    def reset(
        self,
        rng: jax.Array,
        const: CoopForagingConst
    ) -> tuple[CoopForagingState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[i:i+1] for i in range(self.n_agent)]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: CoopForagingConst,
        state: CoopForagingState,
        action: Sequence[Action]
    ) -> tuple[CoopForagingState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.concatenate([agent_action['all'] for agent_action in action])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[i:i+1] for i in range(self.n_agent)]
        reward_lst = [env_reward[i:i+1] for i in range(self.n_agent)]
        done_lst = [env_done[i:i+1] for i in range(self.n_agent)]
        return state, obs_lst, reward_lst, done_lst, info
    
    def export_gif(
        self,
        states: list[CoopForagingState],
        agent_id: int = 0,
        filename: str = "goal_cycle.gif",
        frame_duration_ms: int = 500,
    ):
        self.visualizer.export_gif(states=states, agent_id=agent_id, filename=filename, frame_duration_ms=frame_duration_ms)

    def export_predictive_comparison_gif(
        self,
        states: Sequence[CoopForagingState],
        predicted_next_obs_seq: Sequence[Observation],
        agent_id: int = 0,
        filename: str = "goal_cycle_predictive_comparison.gif",
        frame_duration_ms: int = 500,
    ) -> None:
        self.visualizer.export_predictive_comparison_gif(
            states=states,
            predicted_next_obs_seq=predicted_next_obs_seq,
            agent_id=agent_id,
            filename=filename,
            frame_duration_ms=frame_duration_ms,
        )

    