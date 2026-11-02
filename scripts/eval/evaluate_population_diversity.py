"""How differently the members of a population act in the same states (paper Table 2, Overcooked).

States: self-play trajectories of every member on each of the five original layouts, from the
layout's fixed start state, both seats. Every member then re-encodes each trajectory prefix from its
own zero carry (so pi_j(.|s) is defined for recurrent policies) and the script reports, per layout:

    JSD_pop = mean_s Delta H(s),  Delta H(s) = H(pi_bar(.|s)) - mean_j H(pi_j(.|s))   (nats, <= log 6)
    pairwise JSD matrix d_ij = mean_s JSD(pi_i(.|s) || pi_j(.|s)),  min JSD_pair = min_{i<j} d_ij

With `shared_state_buffer` (the paper setting) the states are pooled from the self-play of every
population listed in `runs`, so all populations are scored on identical states. The paper's CEC numbers
pooled CEC with four baseline populations (not part of this repository), so scoring CEC alone does not
reproduce Table 2 exactly.

Writes to <run>/eval/<eval_name>/: table2.csv (layout, jsd_pop, min_jsd_pair, n_steps, incl. an 'all'
row), entropy_gap_by_layout.csv, entropy_gap_by_layout_source.csv, entropy_gap_vs_timestep.csv,
pairwise_jsd_by_layout.csv, selfplay_returns.csv, heatmap_{delta_h,count}_<layout>.npy,
pairwise_jsd_<layout>.npy and summary.json.

    python scripts/eval/evaluate_population_diversity.py runs=[runs/oc_cec]
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cecp.envs.spaces import Observation
from cecp.eval.common import (
    PolicyLoader, check_same_env, config_from_hydra, eval_output_dir, make_eval_env, model_signature, write_csv,
)
from cecp.eval.diversity import (
    NUM_SEATS, LayoutAccumulator, assert_entropy_gap_bounds, assert_pairwise_jsd_bounds, entropy_gap_from_probs,
    mean_sem, pairwise_jsd_from_probs, safe_div,
)
from cecp.eval.overcooked import GRID_SIZE, ORIGINAL_LAYOUTS, fixed_initial_state
from cecp.eval.rollouts import aot_compile, make_population_replay, make_selfplay_collector
from cecp.runs import RunDir

# Full fp32 matmuls: with TF32 the online (one step) and replay (whole sequence) paths pick different
# kernels and disagree by up to ~2e-2 in probability, which swamps the replay-vs-online check.
jax.config.update('jax_default_matmul_precision', 'float32')
WARN_REPLAY_VS_ONLINE = 1e-3


@dataclass
class PopulationDiversityConfig:
    runs: list[str] = field(default_factory=list)  # populations (e.g. CEC runs); all must share one env
    eval_name: str = 'population_diversity'
    overwrite: bool = False
    ckpt: str = 'best'
    max_agents: int | None = 8              # first N agents of each population (Delta H's maximum depends on N)
    layouts: list[str] = field(default_factory=lambda: list(ORIGINAL_LAYOUTS))
    shared_state_buffer: bool = True        # score every population on the pooled states of all of them
    episodes_per_source: int = 8            # self-play episodes per (population, member, layout)
    episode_length: int | None = None       # None -> the runs' general.game_length
    replay_batch_size: int = 16             # must divide 2 * episodes_per_source
    seed: int | None = None                 # None -> the first run's general.seed
    eval_seed_offset: int = int(4e8)
    env_overrides: dict = field(default_factory=dict)


@dataclass
class Segment:
    """One member's self-play trajectories on one layout, flattened over (episode, seat)."""
    owner: int                  # index of the population (in cfg.runs) that generated it
    source_agent_id: int
    obs: np.ndarray             # [B, T, H, W, C]
    pos: np.ndarray             # [B, T, 2] (row, col)
    online_probs: np.ndarray    # [B, T, A] the acting member's own distribution
    team_return_mean: float
    team_return_std: float

    @property
    def label(self) -> str:
        return f'{self.owner}-a{self.source_agent_id}'


class PopulationScorer:
    def __init__(self, cfg: PopulationDiversityConfig, runs: list[RunDir]) -> None:
        self.cfg = cfg
        check_same_env(runs, cfg.env_overrides)
        lengths = {int(r.config['general']['game_length']) for r in runs}
        if cfg.episode_length is None and len(lengths) > 1:
            raise ValueError(f'Runs have different general.game_length {sorted(lengths)}; set episode_length.')
        self.ev = make_eval_env(runs[0], cfg.env_overrides, cfg.episode_length)
        if self.ev.env_mode != 'overcooked':
            raise ValueError(f'evaluate_population_diversity supports Overcooked only, got {self.ev.env_mode}.')
        if (2 * cfg.episodes_per_source) % cfg.replay_batch_size:
            raise ValueError(f'replay_batch_size={cfg.replay_batch_size} must divide 2 * episodes_per_source.')
        self.starts = {name: fixed_initial_state(self.ev.env, name) for name in cfg.layouts}
        loader = PolicyLoader()
        self.populations = [loader.load_population(r, cfg.ckpt if cfg.ckpt == 'best' else int(cfg.ckpt), max_agents=cfg.max_agents)
                            for r in runs]
        for run, pop in zip(runs, self.populations):
            if len(pop) < 2:
                raise ValueError(f'{run.path} has {len(pop)} agent(s); Delta H needs a population of at least 2.')
        self.stacked = [jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[p.model_state for p in pop]) for pop in self.populations]
        self._collectors: dict[tuple, Any] = {}
        self._replays: dict[tuple, Any] = {}

    def _collector(self, policy) -> Any:
        key = model_signature(policy.model_fn, policy.model_state)
        if key not in self._collectors:
            state0, obs0 = next(iter(self.starts.values()))
            fn = make_selfplay_collector(self.ev.env, self.ev.game_length, self.cfg.episodes_per_source, policy.model_fn)
            self._collectors[key] = aot_compile(fn, jax.random.key(0), policy.model_state, state0, obs0, name='self-play collector')
        return self._collectors[key]

    def collect(self, owner: int, layout: str, rng: jax.Array) -> tuple[list[Segment], jax.Array]:
        """Self-play segments of every member of population `owner` on `layout`."""
        state, obs = self.starts[layout]
        n_seq, length = 2 * self.cfg.episodes_per_source, self.ev.game_length
        segments = []
        for policy in tqdm(self.populations[owner], desc=f'{layout} self-play (population {owner})', leave=False):
            rng, rng_collect = jax.random.split(rng)
            obs_seq, probs, pos, _, _, done, returns = map(np.asarray, self._collector(policy)(rng_collect, policy.model_state, state, obs))
            if done[:, :, :-1].any():
                raise AssertionError(f'{layout}: an episode ended before the last recorded step.')
            pos = pos.reshape(n_seq, length, 2)
            if pos.min() < 0 or pos.max() >= GRID_SIZE:
                raise AssertionError(f'{layout}: agent position outside the {GRID_SIZE}x{GRID_SIZE} grid.')
            team = returns.mean(axis=1)
            segments.append(Segment(owner, policy.agent_id, obs_seq.reshape(n_seq, length, *obs_seq.shape[3:]), pos,
                                    probs.reshape(n_seq, length, -1), float(team.mean()), float(team.std())))
        return segments, rng

    def replay(self, owner: int, obs: np.ndarray) -> np.ndarray:
        """[N, B, T, A] probabilities of every member of population `owner` along the sequences `obs`."""
        policy, stacked, bs = self.populations[owner][0], self.stacked[owner], self.cfg.replay_batch_size
        key = (model_signature(policy.model_fn, policy.model_state), len(self.populations[owner]))
        if key not in self._replays:
            self._replays[key] = aot_compile(make_population_replay(policy.model_fn), jax.random.key(0), stacked,
                                             Observation({'grid_2d': jnp.asarray(obs[:bs])}), name='population replay')
        # One fixed replay key for every call, so identical policies give bitwise-identical probabilities.
        return np.concatenate([np.asarray(self._replays[key](jax.random.key(0), stacked, Observation({'grid_2d': jnp.asarray(obs[s:s + bs])})))
                               for s in range(0, obs.shape[0], bs)], axis=1)

    def score(self, owner: int, segments_by_layout: dict[str, list[Segment]]) -> tuple[dict, list, dict]:
        """Accumulate Delta H / pairwise JSD of population `owner` over the given state segments."""
        agent_ids = [p.agent_id for p in self.populations[owner]]
        accs, return_rows = {}, []
        validation = {'max_replay_vs_online_prob_error': 0.0, 'delta_h_max': 0.0, 'pairwise_jsd_max': 0.0}
        for layout, segments in segments_by_layout.items():
            acc = LayoutAccumulator([s.label for s in segments], self.ev.game_length, agent_ids, GRID_SIZE)
            for src, seg in enumerate(tqdm(segments, desc=f'{layout} replay (population {owner})', leave=False)):
                probs = self.replay(owner, seg.obs)
                if seg.owner == owner:   # the acting member's replayed distribution must match its online one
                    err = float(np.max(np.abs(probs[agent_ids.index(seg.source_agent_id)] - seg.online_probs)))
                    validation['max_replay_vs_online_prob_error'] = max(validation['max_replay_vs_online_prob_error'], err)
                    if err > WARN_REPLAY_VS_ONLINE:
                        print(f'WARNING: {layout}/{seg.label}: replay differs from the online forward by {err:.3e}.')
                    return_rows.append({'layout': layout, 'source_agent_id': seg.source_agent_id,
                                        'mean_return': seg.team_return_mean, 'std_return': seg.team_return_std})
                stats = entropy_gap_from_probs(probs)
                assert_entropy_gap_bounds(stats, probs.shape[-1], f'{layout}/{seg.label}')
                pair_jsd = pairwise_jsd_from_probs(probs)
                assert_pairwise_jsd_bounds(pair_jsd, f'{layout}/{seg.label}')
                validation['delta_h_max'] = max(validation['delta_h_max'], float(stats['delta_h'].max()))
                validation['pairwise_jsd_max'] = max(validation['pairwise_jsd_max'], float(pair_jsd.max()))
                acc.add(src, stats, seg.pos, pair_jsd)
            accs[layout] = acc
        return accs, return_rows, validation


def write_outputs(out_dir, accs: dict[str, LayoutAccumulator], agent_ids, return_rows, validation, extra_summary) -> dict:
    layout_rows, source_rows, timestep_rows, pair_rows, table_rows = [], [], [], [], []
    for layout, acc in accs.items():
        n = int(acc.src_count.sum())
        dh_mean, dh_sem = mean_sem(float(acc.src_dh_sum.sum()), float(acc.src_dh_sqsum.sum()), n)
        jsd = acc.pair_jsd_sum.sum(0) / n
        assert_pairwise_jsd_bounds(jsd, f'{layout} (pooled)')
        iu = np.triu_indices(len(agent_ids), k=1)
        layout_rows.append({'layout': layout, 'n_steps': n, 'delta_h_mean_nats': dh_mean, 'delta_h_sem_nats': dh_sem,
                            'h_bar_mean_nats': float(acc.src_hbar_sum.sum() / n), 'h_mean_nats': float(acc.src_hmean_sum.sum() / n),
                            'delta_h_min': float(acc.src_delta_h_min.min()), 'delta_h_max': float(acc.src_delta_h_max.max())})
        table_rows.append({'layout': layout, 'jsd_pop': dh_mean, 'min_jsd_pair': float(jsd[iu].min()), 'n_steps': n})
        pair_rows += [{'layout': layout, 'agent_i': agent_ids[i], 'agent_j': agent_ids[j], 'jsd_nats': float(jsd[i, j]), 'n_steps': n}
                      for i, j in zip(*iu)]
        for src, label in enumerate(acc.source_labels):
            m, s = mean_sem(float(acc.src_dh_sum[src]), float(acc.src_dh_sqsum[src]), int(acc.src_count[src]))
            source_rows.append({'layout': layout, 'source': label, 'n_steps': int(acc.src_count[src]),
                                'delta_h_mean_nats': m, 'delta_h_sem_nats': s})
        t_mean = safe_div(acc.t_dh_sum.sum(0), acc.t_count.sum(0))
        timestep_rows += [{'layout': layout, 't': t, 'delta_h_mean_nats': float(v), 'count': int(c)}
                          for t, (v, c) in enumerate(zip(t_mean, acc.t_count.sum(0)))]
        np.save(out_dir / f'heatmap_delta_h_{layout}.npy', safe_div(acc.heat_dh_sum.sum(0), acc.heat_count.sum(0)))
        np.save(out_dir / f'heatmap_count_{layout}.npy', acc.heat_count.sum(0))
        np.save(out_dir / f'pairwise_jsd_{layout}.npy', jsd)

    # Across layouts: Delta H pooled over all scored steps; the pair matrix likewise (step-weighted).
    n_all = sum(int(a.src_count.sum()) for a in accs.values())
    dh_all, dh_all_sem = mean_sem(sum(float(a.src_dh_sum.sum()) for a in accs.values()),
                                  sum(float(a.src_dh_sqsum.sum()) for a in accs.values()), n_all)
    jsd_all = sum(a.pair_jsd_sum.sum(0) for a in accs.values()) / n_all
    table_rows.append({'layout': 'all', 'jsd_pop': dh_all,
                       'min_jsd_pair': float(jsd_all[np.triu_indices(len(agent_ids), k=1)].min()), 'n_steps': n_all})

    write_csv(out_dir / 'table2.csv', table_rows)
    write_csv(out_dir / 'entropy_gap_by_layout.csv', layout_rows)
    write_csv(out_dir / 'entropy_gap_by_layout_source.csv', source_rows)
    write_csv(out_dir / 'entropy_gap_vs_timestep.csv', timestep_rows)
    write_csv(out_dir / 'pairwise_jsd_by_layout.csv', pair_rows)
    if return_rows:
        write_csv(out_dir / 'selfplay_returns.csv', return_rows)
    summary = {**extra_summary, 'agent_ids': agent_ids, 'delta_h_mean_nats': dh_all, 'delta_h_sem_nats': dh_all_sem,
               'per_layout': {r['layout']: r for r in layout_rows}, 'validation': validation}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_population_diversity')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(PopulationDiversityConfig, hydra_cfg)
    if not cfg.runs:
        raise ValueError('Pass the population run directories, e.g. runs=[runs/oc_cec]')
    unknown = sorted(set(cfg.layouts) - set(ORIGINAL_LAYOUTS))
    if unknown:
        raise ValueError(f'Unknown layouts {unknown}; expected a subset of {list(ORIGINAL_LAYOUTS)}.')
    runs = [RunDir.open(p) for p in cfg.runs]
    out_dirs = [eval_output_dir(r, cfg.eval_name, asdict(cfg) | {'scored_run': str(r.path)}, cfg.overwrite) for r in runs]
    scorer = PopulationScorer(cfg, runs)
    seed = int(runs[0].config['general']['seed']) if cfg.seed is None else cfg.seed

    shared = None
    if cfg.shared_state_buffer:
        rng = jax.random.key(seed + cfg.eval_seed_offset + 1)
        shared = {layout: [] for layout in cfg.layouts}
        for layout in cfg.layouts:
            for owner in range(len(runs)):
                segments, rng = scorer.collect(owner, layout, rng)
                shared[layout] += segments

    for owner, (run, out_dir) in enumerate(zip(runs, out_dirs)):
        if shared is None:
            rng, own = jax.random.key(seed + cfg.eval_seed_offset), {}
            for layout in cfg.layouts:
                own[layout], rng = scorer.collect(owner, layout, rng)
        accs, return_rows, validation = scorer.score(owner, shared if shared is not None else own)
        summary = write_outputs(out_dir, accs, [p.agent_id for p in scorer.populations[owner]], return_rows, validation, {
            'run': str(run.path), 'state_buffer': 'shared' if shared is not None else 'own',
            'state_sources': [str(r.path) for r in runs] if shared is not None else [str(run.path)],
            'episodes_per_source': cfg.episodes_per_source, 'episode_length': scorer.ev.game_length, 'seed': seed})
        print(f'{run.name}: JSD_pop (Delta H) over all layouts {summary["delta_h_mean_nats"]:.4f} nats -> {out_dir}')


if __name__ == '__main__':
    main()
