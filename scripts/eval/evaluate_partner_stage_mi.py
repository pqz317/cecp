"""How much an ego's action depends on its partner's training stage: I(a; z | s) (paper Fig 18; feeds
Fig 11/19 via visualize_partner_stage_mi_states.py; Overcooked).

    a = the ego's action distribution, z = the partner's stage (early / mid / late checkpoint of a CEC
    population), s = a state, identified by a 96-bit hash of the ego's observation.

    I(a; z | s) = sum_s p(s) * JSD(p(a|s,early), p(a|s,mid), p(a|s,late))

The per-state JSD is the uniform-weight generalized JSD of the three stage-conditional mean action
distributions. A (state, stage) cell qualifies with more than `min_samples_per_cell` samples, a state
when all three of its cells do. Two weightings are reported: `weighted` (p(s) = the state's share of
qualifying samples) and `uniform` (plain mean over qualifying states, the paper figure). The plug-in
estimator is biased upward, so each is compared with a permutation null: the episode -> stage map is
shuffled and the whole pipeline (filters, weighting) re-run `n_permutations` times.

Collection: for each ego agent and layout (a "shard"), the ego (seat 0) plays `episodes_per_partner`
episodes of `episode_length` steps with every partner from the layout's fixed start state; steps with
t <= min_timestep are dropped (every episode starts in the same state). The env's max_steps stays at
the training game length, so the urgency observation channel (last 40 steps) never fires in the window.

Writes to <run>/eval/<eval_name>/:
  partner_stage_mi_by_ego.csv            one row per (layout, ego agent): true / null mean / null std / excess
                                         for both weightings, state and row counts
  partners.csv                           the partner pool and stages
  <layout>/agent_<i>/{raw_rows.npz, states.npz, aggregates.npz, null.npy, summary.json}

    python scripts/eval/evaluate_partner_stage_mi.py runs=[runs/oc_cec,runs/oc_cecp]
    python scripts/plot/plot_partner_stage_mi_bar.py runs/oc_cecp
"""
import functools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PartnerPoolConfig, PolicyLoader, config_from_hydra, eval_env_const, eval_output_dir, flatten_obs, make_eval_env,
    model_signature, resolve_partner_pool, stage_names, write_csv,
)
from cecp.eval.diversity import assert_entropy_gap_bounds, entropy_gap_from_probs
from cecp.eval.overcooked import (
    ORIGINAL_LAYOUTS, check_tiled_state, fixed_initial_state, hash_multipliers, hash_obs, structured_view,
    tile_initial_state,
)
from cecp.eval.rollouts import FixedStartStep, aot_compile, make_fixed_start_ego_collector
from cecp.runs import RunDir

jax.config.update('jax_default_matmul_precision', 'float32')   # see evaluate_population_diversity

NUM_STAGES = 3
METRICS = ('weighted', 'uniform')
# The null is computed in float32 JAX and the reported value in float64 numpy, so the identity-
# permutation check cannot be exact.
TOL_NULL_IDENTITY = 1e-5


@dataclass
class PartnerStageMIConfig:
    runs: list[str] = field(default_factory=list)  # ego runs, e.g. [runs/oc_cec, runs/oc_cecp]
    partners: dict = field(default_factory=lambda: {'num_checkpoints_per_agent': NUM_STAGES, 'include_best': False})
    eval_name: str = 'partner_stage_mi'
    overwrite: bool = False
    layouts: list[str] = field(default_factory=lambda: list(ORIGINAL_LAYOUTS))
    max_ego_agents: int = 0                 # 0 = every agent of each ego run
    episodes_per_partner: int = 1000
    episode_length: int = 60
    min_timestep: int = 10                  # rows kept are t > min_timestep
    min_samples_per_cell: int = 10          # a (state, stage) cell qualifies with more samples than this
    n_permutations: int = 1000
    permutation_chunk: int = 25
    analysis_seed: int = 0
    seed: int | None = None                 # None -> each ego run's general.seed
    eval_seed_offset: int = int(7e8)
    save_raw_rows: bool = True              # visualize_partner_stage_mi_states.py needs them
    env_overrides: dict = field(default_factory=dict)


def shard_seed(seed: int, eval_seed_offset: int, layout_idx: int, run_idx: int, agent_id: int) -> int:
    """Distinct per (layout, ego run, agent); seeds both the rollouts and the permutation null."""
    return seed + eval_seed_offset + 1000003 * (layout_idx + 1) + 1009 * (run_idx + 1) + agent_id


# -- collection ----------------------------------------------------------------------------------

def mi_step_output(multipliers):
    def output(s: FixedStartStep):
        raw = s.env_state._state   # env agent index == role (env_agent_to_role is [[0, 1]]); ego is role 0
        return (hash_obs(flatten_obs(s.ego_obs), multipliers), s.ego_probs,
                raw.agent_pos[:, 0, ::-1].astype(jnp.int32), raw.agent_inv[:, 0].astype(jnp.int32),
                raw.agent_dir_idx[:, 0].astype(jnp.int32), s.ego_done)
    return output


@dataclass
class ShardRows:
    """One row per kept (partner, episode, timestep) sample of a shard, plus one descriptor per state."""
    probs: np.ndarray          # [N, A] float16
    episode_id: np.ndarray     # [N] int32, unique across the shard's partners
    state_id: np.ndarray       # [N] int32, shard-local id of the observation hash
    partner_idx: np.ndarray    # [N] int8
    stage: np.ndarray          # [N] int8
    timestep: np.ndarray       # [N] int16
    state_hash: np.ndarray     # [S, 3] uint32
    state_row: np.ndarray      # [S] ego (row, col, inventory, orientation, first timestep) at the state's first occurrence
    state_col: np.ndarray
    state_inv: np.ndarray
    state_dir: np.ndarray
    state_first_t: np.ndarray
    n_episodes: int
    mean_ego_return: float


def collect_shard(cfg: PartnerStageMIConfig, collect, ego_state, partners, start, seed: int) -> ShardRows:
    n_eps = cfg.episodes_per_partner
    kept_t = np.arange(cfg.episode_length)[cfg.min_timestep + 1:]
    chunks = {k: [] for k in ('hash', 'probs', 'episode', 'timestep', 'partner', 'stage', 'pos', 'inv', 'dir')}
    returns_sum = 0.0
    rng = jax.random.key(seed)
    for partner_idx, (pool_partner, partner) in enumerate(tqdm(partners, desc='    rollouts', leave=False)):
        rng, rng_call = jax.random.split(rng)
        (hashes, probs, pos, inv, direction, done), returns = collect(rng_call, ego_state, partner.model_state, *start)
        if np.asarray(done).any():
            raise AssertionError('An episode ended inside the rollout window; raise the env game length.')
        probs = np.asarray(probs)
        if float(np.abs(probs.sum(-1) - 1.0).max()) > 1e-5:
            raise AssertionError('Ego action probabilities are not normalized.')
        keep = slice(cfg.min_timestep + 1, None)   # scan output is [T, E, ...]
        chunks['hash'].append(np.asarray(hashes)[keep].reshape(-1, 3))
        chunks['probs'].append(probs[keep].reshape(-1, probs.shape[-1]))
        chunks['pos'].append(np.asarray(pos)[keep].reshape(-1, 2))
        chunks['inv'].append(np.asarray(inv)[keep].reshape(-1))
        chunks['dir'].append(np.asarray(direction)[keep].reshape(-1))
        chunks['episode'].append(np.tile(np.arange(partner_idx * n_eps, (partner_idx + 1) * n_eps, dtype=np.int32), len(kept_t)))
        chunks['timestep'].append(np.repeat(kept_t.astype(np.int16), n_eps))
        chunks['partner'].append(np.full(len(kept_t) * n_eps, partner_idx, np.int8))
        chunks['stage'].append(np.full(len(kept_t) * n_eps, pool_partner.stage, np.int8))
        returns_sum += float(np.asarray(returns).sum())

    # float16 in memory as on disk, so re-scoring from raw_rows.npz reproduces these numbers exactly.
    probs = np.concatenate(chunks['probs']).astype(np.float16)
    timestep = np.concatenate(chunks['timestep'])
    unique, state_id = np.unique(structured_view(np.concatenate(chunks['hash']).astype(np.uint32)), return_inverse=True)
    state_id = state_id.astype(np.int32)
    # One descriptor per state, from its earliest occurrence.
    order = np.lexsort((timestep.astype(np.int64), state_id.astype(np.int64)))
    heads = order[np.flatnonzero(np.r_[True, state_id[order][1:] != state_id[order][:-1]])]
    first_row = np.empty(len(unique), np.int64)
    first_row[state_id[heads]] = heads
    pos, inv, direction = (np.concatenate(chunks[k]) for k in ('pos', 'inv', 'dir'))
    return ShardRows(probs, np.concatenate(chunks['episode']), state_id, np.concatenate(chunks['partner']),
                     np.concatenate(chunks['stage']), timestep, unique.view(np.uint32).reshape(-1, 3),
                     pos[first_row, 0].astype(np.int16), pos[first_row, 1].astype(np.int16), inv[first_row].astype(np.int16),
                     direction[first_row].astype(np.int16), timestep[first_row].astype(np.int16),
                     len(partners) * n_eps, returns_sum / (len(partners) * n_eps))


# -- estimator -----------------------------------------------------------------------------------

def aggregate(rows: ShardRows) -> tuple[np.ndarray, np.ndarray]:
    """Sums [S, 3, A] and counts [S, 3] of the action distributions per (state, stage) cell."""
    n_states, n_actions = rows.state_hash.shape[0], rows.probs.shape[1]
    idx = rows.state_id.astype(np.int64) * NUM_STAGES + rows.stage.astype(np.int64)
    counts = np.bincount(idx, minlength=n_states * NUM_STAGES).astype(np.float64)
    sums = np.stack([np.bincount(idx, weights=rows.probs[:, a].astype(np.float64), minlength=n_states * NUM_STAGES)
                     for a in range(n_actions)], axis=-1)
    return sums.reshape(n_states, NUM_STAGES, n_actions), counts.reshape(n_states, NUM_STAGES)


def mi_from_aggregates(sums: np.ndarray, counts: np.ndarray, min_samples: int) -> dict:
    qualifying = np.all(counts > min_samples, axis=1)
    if not qualifying.any():
        raise ValueError('No state qualified (> min_samples_per_cell samples for all three stages); '
                         'lower min_samples_per_cell or raise episodes_per_partner.')
    cell_counts = counts[qualifying]
    mean_probs = sums[qualifying] / cell_counts[..., None]
    mean_probs /= mean_probs.sum(axis=-1, keepdims=True)   # undo float16 quantization
    # [S_q, 3, A] -> [3, S_q, 1, A]: the estimator averages over the leading (stage) axis.
    stats = entropy_gap_from_probs(np.transpose(mean_probs, (1, 0, 2))[:, :, None, :])
    assert_entropy_gap_bounds(stats, mean_probs.shape[-1], 'partner-stage JSD')
    delta_h = stats['delta_h'][:, 0]
    weights = cell_counts.sum(axis=1) / cell_counts.sum()
    return {'weighted': float(np.sum(weights * delta_h)), 'uniform': float(delta_h.mean()), 'qualifying': qualifying,
            'n_states_qualifying': int(qualifying.sum()), 'n_rows_qualifying': int(cell_counts.sum()),
            'delta_h': delta_h, 'weights': weights, 'mean_probs': mean_probs, 'cell_counts': cell_counts}


@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def _null_stats(stage_of_episode, cell_state, cell_episode, cell_sum, cell_count, n_states, n_actions, min_samples):
    """[P] permutations of the episode -> stage map -> [P, 2] (weighted, uniform) I(a; z | s), re-running
    the cell filter, state filter and weighting for every permutation."""
    def one(stages):
        idx = cell_state * NUM_STAGES + stages[cell_episode]
        sums = jax.ops.segment_sum(cell_sum, idx, num_segments=n_states * NUM_STAGES).reshape(n_states, NUM_STAGES, n_actions)
        counts = jax.ops.segment_sum(cell_count, idx, num_segments=n_states * NUM_STAGES).reshape(n_states, NUM_STAGES)
        qualifying = jnp.all(counts > min_samples, axis=1)
        p = sums / jnp.where(counts > 0, counts, 1.0)[..., None]
        p = jnp.where(qualifying[:, None, None], p, 1.0 / n_actions)   # non-qualifying states contribute exactly 0
        p = p / jnp.sum(p, axis=-1, keepdims=True)
        p_bar = jnp.mean(p, axis=1)
        delta_h = jnp.mean(jnp.sum(jax.scipy.special.xlogy(p, p) - jax.scipy.special.xlogy(p, p_bar[:, None, :]), axis=-1), axis=1)

        def average(w):
            total = jnp.sum(w)
            return jnp.where(total > 0, jnp.sum(w * delta_h) / jnp.where(total > 0, total, 1.0), jnp.nan)
        return jnp.stack([average(jnp.where(qualifying, counts.sum(axis=1), 0.0)), average(qualifying.astype(delta_h.dtype))])
    return jax.vmap(one)(stage_of_episode)


def null_distribution(rows: ShardRows, cfg: PartnerStageMIConfig, mi_true: np.ndarray, seed: int) -> np.ndarray:
    """[n_permutations, 2] null of (weighted, uniform)."""
    # Collapse rows to (state, episode) cells once: a permutation only relabels whole episodes.
    key = np.stack([rows.state_id.astype(np.int64), rows.episode_id.astype(np.int64)], axis=1)
    _, cell_id = np.unique(structured_view(key), return_inverse=True)
    n_cells = int(cell_id.max()) + 1
    cell_sum = np.stack([np.bincount(cell_id, weights=rows.probs[:, a].astype(np.float64), minlength=n_cells)
                         for a in range(rows.probs.shape[1])], axis=-1)
    cell_state, cell_episode = np.zeros(n_cells, np.int32), np.zeros(n_cells, np.int32)
    cell_state[cell_id], cell_episode[cell_id] = rows.state_id, rows.episode_id
    # Pad the state count to a power of two so shards of similar size share one compilation; padded
    # states have no samples, never qualify and contribute nothing.
    n_states = 1 << int(np.ceil(np.log2(max(rows.state_hash.shape[0], 1))))
    args = (jnp.asarray(cell_state), jnp.asarray(cell_episode), jnp.asarray(cell_sum, jnp.float32),
            jnp.asarray(np.bincount(cell_id, minlength=n_cells), jnp.float32),
            n_states, rows.probs.shape[1], cfg.min_samples_per_cell)

    identity = np.full(rows.n_episodes, -1, np.int32)
    identity[rows.episode_id] = rows.stage
    check = np.asarray(_null_stats(jnp.asarray(identity)[None], *args)[0], np.float64)
    if np.abs(check - mi_true).max() > TOL_NULL_IDENTITY:
        raise AssertionError(f'Permutation pipeline gives {check} under the identity permutation, estimate is {mi_true}.')
    rng = np.random.default_rng(cfg.analysis_seed + seed)
    null = np.empty((cfg.n_permutations, len(METRICS)))
    for start in tqdm(range(0, cfg.n_permutations, cfg.permutation_chunk), desc='    permutations', leave=False):
        size = min(cfg.permutation_chunk, cfg.n_permutations - start)
        null[start:start + size] = np.asarray(_null_stats(jnp.asarray(np.stack([rng.permutation(identity) for _ in range(size)])), *args))
    return null


def write_shard(shard_dir: Path, rows: ShardRows, result: dict, null: np.ndarray, summary: dict, save_raw_rows: bool) -> None:
    shard_dir.mkdir(parents=True)
    if save_raw_rows:
        np.savez_compressed(shard_dir / 'raw_rows.npz', probs=rows.probs, episode_id=rows.episode_id, state_id=rows.state_id,
                            partner_idx=rows.partner_idx, stage=rows.stage, timestep=rows.timestep)
    np.savez_compressed(shard_dir / 'states.npz', state_hash=rows.state_hash, ego_row=rows.state_row, ego_col=rows.state_col,
                        ego_inv=rows.state_inv, ego_dir=rows.state_dir, first_timestep=rows.state_first_t)
    np.savez_compressed(shard_dir / 'aggregates.npz', state_id=np.flatnonzero(result['qualifying']).astype(np.int32),
                        mean_probs=result['mean_probs'].astype(np.float32), num_samples=result['cell_counts'].astype(np.int32),
                        delta_h=result['delta_h'], weights=result['weights'])
    np.save(shard_dir / 'null.npy', null)
    (shard_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


# -- driver --------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_partner_stage_mi')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(PartnerStageMIConfig, hydra_cfg)
    if not cfg.runs:
        raise ValueError('Pass the ego run directories, e.g. runs=[runs/oc_cec,runs/oc_cecp]')
    if cfg.min_timestep >= cfg.episode_length - 1:
        raise ValueError(f'min_timestep={cfg.min_timestep} leaves no rows in {cfg.episode_length}-step episodes.')
    loader = PolicyLoader()
    for run_idx, path in enumerate(cfg.runs):
        run = RunDir.open(path)
        pool, selection = resolve_partner_pool(PartnerPoolConfig(**cfg.partners), ego_run=run)
        if any(p.stage is None for p in pool) or sorted({p.stage for p in pool}) != list(range(NUM_STAGES)):
            raise ValueError(f'The partner pool needs exactly {NUM_STAGES} stages and no best checkpoints, got {selection}.')
        counts = np.bincount([p.stage for p in pool], minlength=NUM_STAGES)
        if not np.all(counts == counts[0]):
            raise AssertionError(f'Partner pool is unbalanced across stages: {counts.tolist()}.')
        partners = [(p, loader.load_entry(p.entry, run)) for p in pool]
        if len({model_signature(q.model_fn, q.model_state) for _, q in partners}) != 1:
            raise ValueError('All partners must share one architecture (one compiled collector serves them all).')

        ev = make_eval_env(run, cfg.env_overrides)
        if ev.env_mode != 'overcooked':
            raise ValueError(f'evaluate_partner_stage_mi supports Overcooked only, got {ev.env_mode}.')
        if cfg.episode_length > ev.game_length:
            raise ValueError(f'episode_length={cfg.episode_length} exceeds the env game length {ev.game_length}.')
        eval_env = FrozenAssignmentEnv(ev.env, cfg.episodes_per_partner, env_agent_to_role=[[0, 1]] * cfg.episodes_per_partner)
        env_const = eval_env_const(eval_env, ev.env_mode, False)
        out_dir = eval_output_dir(run, cfg.eval_name, asdict(cfg) | {'run_idx': run_idx, 'partner_selection': selection}, cfg.overwrite)
        names = stage_names(NUM_STAGES)
        write_csv(out_dir / 'partners.csv', [{'partner_idx': i, 'partner_name': p.name, 'run': p.entry.run, 'agent_id': p.entry.agent_id,
                                              'ckpt': p.entry.ckpt, 'stage': p.stage, 'stage_name': names[p.stage]} for i, (p, _) in enumerate(partners)])
        egos = loader.load_population(run)
        if cfg.max_ego_agents > 0:
            egos = egos[:cfg.max_ego_agents]
        seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed

        rows_out, compiled = [], {}
        for layout_idx, layout in enumerate(cfg.layouts):
            start = tile_initial_state(*fixed_initial_state(ev.env, layout), cfg.episodes_per_partner)
            check_tiled_state(*start, eval_env, env_const)
            obs_dim = int(flatten_obs(start[1][0]).shape[-1])
            for ego in egos:
                key = model_signature(ego.model_fn, ego.model_state)
                if key not in compiled:
                    fn = make_fixed_start_ego_collector(eval_env, cfg.episode_length, env_const, ego.model_fn, partners[0][1].model_fn,
                                                        mi_step_output(hash_multipliers(obs_dim)))
                    compiled[key] = aot_compile(fn, jax.random.key(0), ego.model_state, partners[0][1].model_state, *start, name='stage collector')
                s_seed = shard_seed(seed, cfg.eval_seed_offset, layout_idx, run_idx, ego.agent_id)
                print(f'[{layout}] {run.name} agent {ego.agent_id}: collecting')
                t0 = time.perf_counter()
                rows = collect_shard(cfg, compiled[key], ego.model_state, partners, start, s_seed)
                result = mi_from_aggregates(*aggregate(rows), cfg.min_samples_per_cell)
                mi_true = np.array([result['weighted'], result['uniform']])
                null = null_distribution(rows, cfg, mi_true, s_seed)
                row = {'layout': layout, 'run': str(run.path), 'label': run.name, 'ego_agent_id': ego.agent_id}
                for k, metric in enumerate(METRICS):
                    row |= {f'mi_{metric}_true': float(mi_true[k]), f'mi_{metric}_null_mean': float(null[:, k].mean()),
                            f'mi_{metric}_null_std': float(null[:, k].std(ddof=1)) if len(null) > 1 else 0.0,
                            f'mi_{metric}_excess': float(mi_true[k] - null[:, k].mean())}
                row |= {'n_rows': len(rows.probs), 'n_states_total': len(rows.state_hash),
                        'n_states_qualifying': result['n_states_qualifying'],
                        'frac_rows_retained': result['n_rows_qualifying'] / len(rows.probs), 'mean_ego_return': rows.mean_ego_return}
                summary = {**row, 'shard_seed': s_seed, 'num_partners': len(partners), 'metric_names': list(METRICS),
                           'episodes_per_partner': cfg.episodes_per_partner, 'wall_clock_s': time.perf_counter() - t0}
                write_shard(out_dir / layout / f'agent_{ego.agent_id}', rows, result, null, summary, cfg.save_raw_rows)
                rows_out.append(row)
                print(f'    uniform I={row["mi_uniform_true"]:.5f} (shuffle {row["mi_uniform_null_mean"]:.5f}), weighted '
                      f'I={row["mi_weighted_true"]:.5f} (shuffle {row["mi_weighted_null_mean"]:.5f}); '
                      f'{row["n_states_qualifying"]}/{row["n_states_total"]} states qualify')
        write_csv(out_dir / 'partner_stage_mi_by_ego.csv', rows_out)
        print(f'Wrote {out_dir}')


if __name__ == '__main__':
    main()
