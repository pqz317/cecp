from typing import Literal

from ..config import CoopForagingEnvConfig, EnvConfig, OvercookedEnvConfig

EnvMode = Literal['coop_foraging', 'overcooked']


def make_env(
    env_cfg: EnvConfig,
    num_players: int,
    game_length: int,
    force_eval_on_reserved: bool | None = None,
) -> tuple[object, EnvMode, bool]:
    """Build a Coop Foraging or Overcooked env from its config.

    Returns (env, env_mode, eval_on_reserved), where eval_on_reserved says whether evaluation should use
    the held-out goal positions / layouts. `force_eval_on_reserved` overrides the config's choice.
    """
    if isinstance(env_cfg, CoopForagingEnvConfig):
        from .coop_foraging import CoopForagingEnv
        env = CoopForagingEnv(
            num_agents=num_players,
            max_steps=game_length,
            obstacle_density=env_cfg.obstacle_density,
            heldout_goal_ratio=env_cfg.heldout_goal_ratio,
            heldout_goal_seed=env_cfg.heldout_goal_seed,
            train_layout_mode=env_cfg.train_layout_mode,
            train_procgen_seed=env_cfg.train_procgen_seed,
        )
        eval_on_reserved = env_cfg.eval_on_reserved_goals if force_eval_on_reserved is None else force_eval_on_reserved
        if eval_on_reserved and env.num_heldout_goal_positions < env.n_goal:
            raise ValueError(
                f"env.eval_on_reserved_goals requires at least n_goal={env.n_goal} reserved positions, "
                f"but got {env.num_heldout_goal_positions}. Increase env.heldout_goal_ratio."
            )
        return env, 'coop_foraging', eval_on_reserved

    if isinstance(env_cfg, OvercookedEnvConfig):
        if num_players != 2:
            raise ValueError(f"overcooked requires num_players == 2, got {num_players}")
        from .overcooked import OvercookedEnv
        env = OvercookedEnv(
            layout=env_cfg.layout,
            max_steps=game_length,
            pad_obs_shape_to=env_cfg.pad_obs_shape_to,
            heldout_layout_seed=env_cfg.heldout_layout_seed,
            heldout_layout_num_samples=env_cfg.heldout_layout_num_samples,
            train_layout_mode=env_cfg.train_layout_mode,
            train_procgen_seed=env_cfg.train_procgen_seed,
        )
        eval_on_reserved = env_cfg.eval_on_reserved_layouts if force_eval_on_reserved is None else force_eval_on_reserved
        if eval_on_reserved and env.num_heldout_layouts < 1:
            raise ValueError(
                "env.eval_on_reserved_layouts requires heldout layouts, but none were generated. "
                "Increase env.heldout_layout_num_samples."
            )
        return env, 'overcooked', eval_on_reserved

    raise TypeError(f'Unsupported env config type {type(env_cfg).__name__}')
