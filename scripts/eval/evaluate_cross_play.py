"""Cross-play between all agents of a population (paper Fig 6a/15; Fig 12 data for Coop Foraging).

For each run, every pair of agents (and every agent with itself) plays `num_episodes` episodes, one
per environment copy, on the run's evaluation environments: the held-out goal positions (Coop
Foraging) or held-out procedurally generated layouts (Overcooked). For Overcooked runs the pairs also
play the five original layouts from their fixed start states. Team return = mean of the two seats.

Writes to <run>/eval/<eval_name>/:
  pair_metrics.csv          one row per pair: team / agent_i / agent_j return stats
  per_episode_returns.csv   one row per (pair, episode)
  return_matrix.{npy,csv}   mean team return per pair (diagonal = self-play)
  return_matrix_<layout>.{npy,csv}, return_matrix_original_avg.{npy,csv}   Overcooked original layouts
  summary.json

    python scripts/eval/evaluate_cross_play.py runs=[runs/oc_cec,runs/oc_cecp]
    python scripts/plot/plot_cross_play.py runs/oc_cec runs/oc_cecp
"""
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import hydra
import jax
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PolicyLoader, check_same_env, compute_stats, config_from_hydra, eval_env_const, eval_output_dir, make_eval_env,
    model_signature, write_csv, write_matrix_csv,
)
from cecp.eval.rollouts import compile_fixed_start_pair_evaluator, compile_pair_evaluator
from cecp.runs import RunDir


@dataclass
class CrossPlayConfig:
    runs: list[str] = field(default_factory=list)
    eval_name: str = 'cross_play'
    overwrite: bool = False
    ckpt: str = 'best'                      # 'best' or a step
    seed: int | None = None                 # None -> each run's general.seed
    eval_seed_offset: int = int(3e8)
    num_episodes: int = 100                 # episodes (= held-out environments) per pair
    episode_length: int | None = None       # None -> the runs' general.game_length
    env_overrides: dict[str, Any] = field(default_factory=dict)
    eval_original_layouts: bool = True      # Overcooked: also play the five original layouts


class CrossPlayEvaluator:
    """Holds the evaluation envs and compiled functions shared by all runs."""

    def __init__(self, cfg: CrossPlayConfig, runs: list[RunDir]) -> None:
        self.cfg = cfg
        check_same_env(runs, cfg.env_overrides)
        lengths = {int(r.config['general']['game_length']) for r in runs}
        if cfg.episode_length is None and len(lengths) > 1:
            raise ValueError(f'Runs have different general.game_length {sorted(lengths)}; set episode_length.')
        self.ev = make_eval_env(runs[0], cfg.env_overrides, cfg.episode_length)
        if self.ev.num_players != 2:
            raise ValueError(f'Cross-play needs 2-player games, got {self.ev.num_players} players.')
        self.eval_env = FrozenAssignmentEnv(self.ev.env, cfg.num_episodes, env_agent_to_role=[[0, 1]] * cfg.num_episodes)
        self.env_const = eval_env_const(self.eval_env, self.ev.env_mode, self.ev.eval_on_reserved)
        self.original_layouts = self.ev.env_mode == 'overcooked' and cfg.eval_original_layouts
        self.layout_starts: list[tuple[str, Any, list]] = []
        if self.original_layouts:
            from cecp.eval.overcooked import ORIGINAL_LAYOUTS, fixed_initial_state
            self.layout_starts = [(name, *fixed_initial_state(self.ev.env, name)) for name in ORIGINAL_LAYOUTS]
        self._compiled: dict[tuple, Any] = {}

    def _compile(self, model_fn: Any, model_state: Any) -> tuple[Any, Any]:
        key = model_signature(model_fn, model_state)
        if key not in self._compiled:
            pair_fn = compile_pair_evaluator(self.eval_env, self.ev.game_length, self.env_const,
                                             model_fn, model_fn, model_state, model_state)
            layout_fn = None
            if self.original_layouts:
                _, state0, obs0 = self.layout_starts[0]
                layout_fn = compile_fixed_start_pair_evaluator(self.ev.env, self.ev.game_length, self.cfg.num_episodes,
                                                               model_fn, model_fn, model_state, model_state, state0, obs0)
            self._compiled[key] = (pair_fn, layout_fn)
        return self._compiled[key]

    def evaluate_run(self, run: RunDir, loader: PolicyLoader) -> None:
        cfg = self.cfg
        ckpt = cfg.ckpt if cfg.ckpt == 'best' else int(cfg.ckpt)
        out_dir = eval_output_dir(run, cfg.eval_name, asdict(cfg) | {'runs': [str(run.path)]}, cfg.overwrite)
        policies = loader.load_population(run, ckpt)
        agent_ids = [p.agent_id for p in policies]
        states = {p.agent_id: p.model_state for p in policies}
        pair_fn, layout_fn = self._compile(policies[0].model_fn, policies[0].model_state)

        seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed
        rng_eval = jax.random.key(seed + cfg.eval_seed_offset)
        n = len(agent_ids)
        pos = {agent_id: i for i, agent_id in enumerate(agent_ids)}
        return_matrix = np.full((n, n), np.nan, dtype=np.float32)
        layout_matrices = {name: np.full((n, n), np.nan, dtype=np.float32) for name, *_ in self.layout_starts}
        pair_rows, episode_rows = [], []

        pairs = list(combinations(agent_ids, 2)) + [(a, a) for a in agent_ids]
        for pair_idx, (agent_i, agent_j) in enumerate(tqdm(pairs, desc=f'{run.name} pairs', unit='pair')):
            rng_eval, rng_pair = jax.random.split(rng_eval)
            ret_i, ret_j = (np.asarray(r) for r in pair_fn(rng_pair, states[agent_i], states[agent_j]))
            team = 0.5 * (ret_i + ret_j)
            pair_rows.append({
                'pair_index': pair_idx, 'agent_i': agent_i, 'agent_j': agent_j, 'num_episodes': len(team),
                **{f'team_{k}': v for k, v in compute_stats(team).items()},
                **{f'agent_i_{k}': v for k, v in compute_stats(ret_i).items()},
                **{f'agent_j_{k}': v for k, v in compute_stats(ret_j).items()},
            })
            episode_rows += [{'pair_index': pair_idx, 'agent_i': agent_i, 'agent_j': agent_j, 'episode_index': e,
                              'team_return': float(team[e]), 'agent_i_return': float(ret_i[e]),
                              'agent_j_return': float(ret_j[e])} for e in range(len(team))]
            return_matrix[pos[agent_i], pos[agent_j]] = return_matrix[pos[agent_j], pos[agent_i]] = team.mean()

            # The original-layout episodes continue the same RNG chain as the held-out ones (as in the
            # paper runs), so disabling them changes the held-out results of every later pair.
            for name, init_state, init_obs in self.layout_starts:
                rng_eval, rng_layout = jax.random.split(rng_eval)
                l_i, l_j = layout_fn(rng_layout, states[agent_i], states[agent_j], init_state, init_obs)
                mean = float(np.mean(0.5 * (np.asarray(l_i) + np.asarray(l_j))))
                layout_matrices[name][pos[agent_i], pos[agent_j]] = layout_matrices[name][pos[agent_j], pos[agent_i]] = mean

        write_csv(out_dir / 'pair_metrics.csv', pair_rows)
        write_csv(out_dir / 'per_episode_returns.csv', episode_rows)
        np.save(out_dir / 'return_matrix.npy', return_matrix)
        write_matrix_csv(out_dir / 'return_matrix.csv', return_matrix, agent_ids)
        for name, matrix in layout_matrices.items():
            np.save(out_dir / f'return_matrix_{name}.npy', matrix)
            write_matrix_csv(out_dir / f'return_matrix_{name}.csv', matrix, agent_ids)
        if layout_matrices:
            original_avg = np.nanmean(np.stack(list(layout_matrices.values())), axis=0)
            np.save(out_dir / 'return_matrix_original_avg.npy', original_avg)
            write_matrix_csv(out_dir / 'return_matrix_original_avg.csv', original_avg, agent_ids)

        # Headline numbers use cross pairs only (the diagonal is self-play).
        cross_team = np.array([r['team_return'] for r in episode_rows if r['agent_i'] != r['agent_j']], dtype=np.float32)
        summary = {
            'run': str(run.path), 'ckpt': cfg.ckpt, 'agent_ids': agent_ids, 'num_pairs': len(pairs),
            'num_episodes_per_pair': cfg.num_episodes, 'episode_length': self.ev.game_length, 'seed': seed,
            'env_mode': self.ev.env_mode, 'eval_on_reserved': self.ev.eval_on_reserved,
            'original_layouts': list(layout_matrices),
            'cross_play_team_return': compute_stats(cross_team) if len(cross_team) else None,
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        if len(cross_team):
            print(f'{run.name}: cross-play team return {cross_team.mean():.3f} (std {cross_team.std():.3f}) -> {out_dir}')


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_cross_play')
def main(cfg: DictConfig) -> None:
    cfg = config_from_hydra(CrossPlayConfig, cfg)
    if not cfg.runs:
        raise ValueError('Pass the run directories to evaluate, e.g. runs=[runs/oc_cec,runs/oc_cecp]')
    runs = [RunDir.open(p) for p in cfg.runs]
    evaluator = CrossPlayEvaluator(cfg, runs)
    loader = PolicyLoader()
    for run in runs:
        evaluator.evaluate_run(run, loader)


if __name__ == '__main__':
    main()
