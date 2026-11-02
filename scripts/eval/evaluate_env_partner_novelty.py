"""Familiar vs novel environments x familiar vs novel partners (paper Fig 4, Coop Foraging).

For a CEC population run: the familiar partner is the agent itself (self-play), novel partners are the
other agents of the population. For a CECP run (one with partners.yaml): familiar partners are the
'best' checkpoints of the partner agents it trained with, novel partners those of the held-out partner
agents. Familiar environments are the training goal positions, novel ones the held-out goal positions
(or, for runs trained on a single fixed procedurally generated environment, another seed's environment).
Each agent's score is its mean team return over its partners; the run's score is the mean over agents.

Writes to <run>/eval/<eval_name>/: results.csv (partner_novelty, env_novelty, mean, std, n_agents),
per_agent_returns.csv and summary.json.

    python scripts/eval/evaluate_env_partner_novelty.py runs=[runs/cf_cec,runs/cf_cecp]
    python scripts/plot/plot_env_partner_novelty.py runs/cf_cec runs/cf_cecp
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
    LoadedPolicy, PolicyLoader, config_from_hydra, eval_env_const, eval_output_dir, make_eval_env, model_signature,
    write_csv,
)
from cecp.eval.rollouts import compile_pair_evaluator
from cecp.runs import PARTNERS_FILE, RunDir

NOVELTY = ('familiar', 'novel')


@dataclass
class EnvPartnerNoveltyConfig:
    runs: list[str] = field(default_factory=list)
    eval_name: str = 'env_partner_novelty'
    overwrite: bool = False
    seed: int | None = None                 # None -> each run's general.seed
    eval_seed_offset: int = 0
    num_episodes: int = 32                  # episodes per (agent, partner) pair
    novel_procgen_seed: int = 1             # novel env for runs trained with env.train_layout_mode=fixed_procgen


class NoveltyEvaluator:
    def __init__(self, cfg: EnvPartnerNoveltyConfig) -> None:
        self.cfg = cfg
        self._compiled: dict[tuple, Any] = {}

    def _envs(self, run: RunDir) -> dict[str, tuple[Any, Any, int]]:
        """env novelty -> (batched eval env, env const, episode length)."""
        envs = {}
        for novelty in NOVELTY:
            overrides, reserved = {}, novelty == 'novel'
            if novelty == 'novel' and run.config['env'].get('train_layout_mode') == 'fixed_procgen':
                overrides, reserved = {'train_procgen_seed': self.cfg.novel_procgen_seed}, False
            ev = make_eval_env(run, overrides, force_eval_on_reserved=reserved)
            if ev.env_mode != 'coop_foraging':
                raise ValueError(f'evaluate_env_partner_novelty supports Coop Foraging runs only, got {ev.env_mode}.')
            batched = FrozenAssignmentEnv(ev.env, self.cfg.num_episodes, env_agent_to_role=[[0, 1]] * self.cfg.num_episodes)
            envs[novelty] = (batched, eval_env_const(batched, ev.env_mode, ev.eval_on_reserved), ev.game_length)
        return envs

    def _team_return(self, env: tuple, rng: jax.Array, ego: LoadedPolicy, partner: LoadedPolicy) -> float:
        batched, const, length = env
        key = (id(batched), model_signature(ego.model_fn, ego.model_state), model_signature(partner.model_fn, partner.model_state))
        if key not in self._compiled:
            self._compiled[key] = compile_pair_evaluator(batched, length, const, ego.model_fn, partner.model_fn,
                                                         ego.model_state, partner.model_state)
        ret_i, ret_j = self._compiled[key](rng, ego.model_state, partner.model_state)
        return float(0.5 * (ret_i.mean() + ret_j.mean()))

    def evaluate_run(self, run: RunDir, loader: PolicyLoader) -> None:
        cfg = self.cfg
        out_dir = eval_output_dir(run, cfg.eval_name, asdict(cfg) | {'runs': [str(run.path)]}, cfg.overwrite)
        agents = loader.load_population(run)
        is_cecp = (run.path / PARTNERS_FILE).exists()
        if is_cecp:
            manifest = run.read_partners()
            # As in the paper evaluation: each partner agent's 'best' checkpoint only.
            partners = {
                'familiar': [loader.load_entry(e, run) for e in manifest.train if e.ckpt == 'best'],
                'novel': [loader.load_entry(e, run) for e in manifest.heldout if e.ckpt == 'best'],
            }
            if not partners['novel']:
                raise ValueError(f'{run.path} has no held-out partners (partners.heldout_ratio was 0).')
        elif len(agents) < 2:
            raise ValueError(f'{run.path}: novel-partner evaluation of a population needs at least 2 agents.')

        envs = self._envs(run)
        seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed
        base_rng = jax.random.key(seed + cfg.eval_seed_offset)
        rows, agent_rows = [], []
        condition_idx = 0
        for partner_novelty in NOVELTY:
            for env_novelty in NOVELTY:
                condition_rng = jax.random.fold_in(base_rng, condition_idx)
                condition_idx += 1
                per_agent: dict[int, list[float]] = {a.agent_id: [] for a in agents}
                if is_cecp:
                    pairs = [(ego, partner) for ego in agents for partner in partners[partner_novelty]]
                elif partner_novelty == 'familiar':
                    pairs = [(a, a) for a in agents]
                else:
                    pairs = list(combinations(agents, 2))
                for pair_idx, (ego, partner) in enumerate(tqdm(pairs, desc=f'{run.name} {partner_novelty} partner / {env_novelty} env', leave=False)):
                    team = self._team_return(envs[env_novelty], jax.random.fold_in(condition_rng, pair_idx), ego, partner)
                    per_agent[ego.agent_id].append(team)
                    if not is_cecp and partner.agent_id != ego.agent_id:
                        per_agent[partner.agent_id].append(team)   # a population pair scores both its agents
                agent_means = [float(np.mean(v)) for v in per_agent.values()]
                rows.append({'partner_novelty': partner_novelty, 'env_novelty': env_novelty,
                             'mean': float(np.mean(agent_means)), 'std': float(np.std(agent_means)), 'n_agents': len(agent_means)})
                agent_rows += [{'partner_novelty': partner_novelty, 'env_novelty': env_novelty, 'agent_id': agent_id,
                                'mean_team_return': mean, 'n_partners': len(per_agent[agent_id])}
                               for agent_id, mean in zip(per_agent, agent_means)]
                print(f'{run.name}: {partner_novelty} partner, {env_novelty} env -> {rows[-1]["mean"]:.3f} (std {rows[-1]["std"]:.3f})')

        write_csv(out_dir / 'results.csv', rows)
        write_csv(out_dir / 'per_agent_returns.csv', agent_rows)
        summary = {'run': str(run.path), 'model_type': 'cecp' if is_cecp else 'population', 'seed': seed,
                   'num_episodes': cfg.num_episodes, 'episode_length': envs['familiar'][2],
                   'train_layout_mode': run.config['env'].get('train_layout_mode')}
        if is_cecp:
            summary['num_partners'] = {k: len(v) for k, v in partners.items()}
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_env_partner_novelty')
def main(cfg: DictConfig) -> None:
    cfg = config_from_hydra(EnvPartnerNoveltyConfig, cfg)
    if not cfg.runs:
        raise ValueError('Pass the run directories to evaluate, e.g. runs=[runs/cf_cec,runs/cf_cecp]')
    evaluator, loader = NoveltyEvaluator(cfg), PolicyLoader()
    for path in cfg.runs:
        evaluator.evaluate_run(RunDir.open(path), loader)


if __name__ == '__main__':
    main()
