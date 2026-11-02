"""Render the states where an ego's action depends most on its partner's stage (paper Fig 11, 19).

Takes one shard (layout, ego agent) written by evaluate_partner_stage_mi.py, ranks its qualifying
states by per-state JSD, and draws the top-K as Overcooked frames next to the ego's mean action
distribution against early / mid / late partners.

A state is stored only as a hash of the ego's observation, so it is recovered by replaying the
collection: same fixed start state, same shard seed (read from the shard's summary.json), one RNG
split per partner in partner order, and the very same step body (`make_fixed_start_ego_collector`)
with a different per-step output. The whole episode batch of the visiting partner is replayed (the
batch cannot be subset without changing the sampled actions), and every recovered frame is re-hashed
on device and on the host and checked against the stored hash and ego descriptor.

Writes to <run>/eval/<eval_name>/: state_rank<r>_id<state_id>.png/.svg, top_states.png and
top_states.csv.

    python scripts/eval/visualize_partner_stage_mi_states.py run=runs/oc_cecp layout=forced_coord_9 agent_id=3
"""
import csv
import json
from dataclasses import asdict, dataclass

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from omegaconf import DictConfig

from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PartnerPoolConfig, PolicyLoader, config_from_hydra, eval_env_const, eval_output_dir, flatten_obs, make_eval_env,
    read_eval_config, resolve_partner_pool, stage_names, write_csv,
)
from cecp.eval.display import OC_ORIGINAL_LAYOUT_NAMES, run_label
from cecp.eval.overcooked import (
    ACTION_NAMES, GRID_SIZE, INVENTORY_NAMES, ORIENTATION_NAMES, check_tiled_state, fixed_initial_state,
    hash_multipliers, hash_obs, hash_obs_numpy, render_grid_frame, state_for_rendering, tile_initial_state,
)
from cecp.eval.plotting import save_fig
from cecp.eval.rollouts import FixedStartStep, aot_compile, make_fixed_start_ego_collector
from cecp.runs import RunDir

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'float32')   # as in evaluate_partner_stage_mi

STAGE_COLORS = ('#bcbddc', '#756bb1', '#3f007d')    # early / mid / late
EGO_COLOR, PARTNER_COLOR = '#e03131', '#1c7ed6'
FONTSIZE = 18


@dataclass
class VisualizeStatesConfig:
    run: str = ''                           # the ego run evaluated with evaluate_partner_stage_mi.py
    mi_eval_name: str = 'partner_stage_mi'
    layout: str = 'forced_coord_9'
    agent_id: int = 3
    top_k: int = 10
    min_total_samples: int = 0              # extra filter on a state's total samples over the 3 stages
    eval_name: str | None = None            # None -> partner_stage_mi_states/<layout>/agent_<id>
    overwrite: bool = False


def select_states(shard_dir, cfg: VisualizeStatesConfig, episodes_per_partner: int) -> list[dict]:
    """Top-K qualifying states by JSD, each with one occurrence (partner, env index, t) to replay;
    occurrences from partners already scheduled are preferred, to replay as few partners as possible."""
    agg, states, raw = (np.load(shard_dir / f) for f in ('aggregates.npz', 'states.npz', 'raw_rows.npz'))
    keep = agg['num_samples'].sum(axis=1) >= cfg.min_total_samples
    order = np.flatnonzero(keep)[np.argsort(-agg['delta_h'][keep], kind='stable')][:cfg.top_k]
    if order.size == 0:
        raise ValueError(f'No qualifying state in {shard_dir} has >= {cfg.min_total_samples} samples.')
    scheduled, selected = set(), []
    for rank, q in enumerate(order, start=1):
        sid = int(agg['state_id'][q])
        rows = np.flatnonzero(raw['state_id'] == sid)
        reuse = [r for r in rows if int(raw['partner_idx'][r]) in scheduled]
        pick = reuse[0] if reuse else rows[0]
        partner_idx, episode_id = int(raw['partner_idx'][pick]), int(raw['episode_id'][pick])
        scheduled.add(partner_idx)
        selected.append({'rank': rank, 'state_id': sid, 'delta_h': float(agg['delta_h'][q]), 'weight': float(agg['weights'][q]),
                         'num_samples': agg['num_samples'][q].astype(int), 'mean_probs': agg['mean_probs'][q].astype(np.float64),
                         'state_hash': states['state_hash'][sid].astype(np.uint32),
                         'ego': (int(states['ego_row'][sid]), int(states['ego_col'][sid]), int(states['ego_inv'][sid]), int(states['ego_dir'][sid])),
                         'first_timestep': int(states['first_timestep'][sid]), 'partner_idx': partner_idx, 'episode_id': episode_id,
                         'env_idx': episode_id - partner_idx * episodes_per_partner, 'timestep': int(raw['timestep'][pick])})
    return selected


def replay_output(multipliers):
    def output(s: FixedStartStep):
        obs_flat = flatten_obs(s.ego_obs)
        return hash_obs(obs_flat, multipliers), obs_flat.astype(jnp.uint8), s.env_state
    return output


def draw_frame(ax, frame: np.ndarray, state) -> None:
    ax.imshow(frame, extent=(-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5), interpolation='nearest')
    raw = state._state
    for seat, color, label in ((0, EGO_COLOR, 'EGO'), (1, PARTNER_COLOR, 'P')):
        x, y = int(raw.agent_pos[seat, 0]), int(raw.agent_pos[seat, 1])
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=2.5, zorder=3))
        ax.text(x, y - 0.5, label, color='white', fontsize=7, fontweight='bold', ha='center', va='top', zorder=4,
                bbox=dict(facecolor=color, edgecolor='none', pad=1.0))
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.grid(False)


def plot_state(out_stem, s: dict, frame, state, names, partner_name: str, header: str) -> None:
    fig = plt.figure(figsize=(13.5, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.45], height_ratios=[1.4, 2.6, 0.8], wspace=0.35, top=0.86, bottom=0.08)
    ax_frame, ax_bars = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[1, 1])
    draw_frame(ax_frame, frame, state)
    ax_frame.set_title('ego = red, partner = blue', fontsize=12)
    ax_frame.set_xlabel('x (col)', fontsize=FONTSIZE)
    ax_frame.set_ylabel('y (row)', fontsize=FONTSIZE)
    x = np.arange(s['mean_probs'].shape[1])
    for k, name in enumerate(names):
        ax_bars.bar(x + (k - 1) * 0.26, s['mean_probs'][k], 0.26, color=STAGE_COLORS[k],
                    label=f'{name} partner (n={int(s["num_samples"][k])})')
    ax_bars.set_xticks(x, ACTION_NAMES[:len(x)], fontsize=FONTSIZE)
    ax_bars.tick_params(axis='y', labelsize=12)
    ax_bars.set_ylim(0, 1)
    ax_bars.set_ylabel(r'$E[\pi_{ego}(a \mid s, \mathrm{stage})]$', fontsize=FONTSIZE)
    ax_bars.legend(fontsize=FONTSIZE, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.02))
    ax_bars.grid(axis='y', alpha=0.3)
    row, col, inv, direction = s['ego']
    counts = ', '.join(f'{name}={int(n)}' for name, n in zip(names, s['num_samples']))
    fig.suptitle(f'{header}\nrank {s["rank"]}  state {s["state_id"]}  JSD={s["delta_h"]:.3f} nats  p(s)={s["weight"]:.4f}\n'
                 f'n: {counts}   ego holds {INVENTORY_NAMES.get(inv, inv)}, faces {ORIENTATION_NAMES[direction]}   '
                 f'frame from partner {s["partner_idx"]} ({partner_name}), episode {s["episode_id"]}, t={s["timestep"]}', fontsize=9, y=0.99)
    save_fig(fig, out_stem)


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='visualize_partner_stage_mi_states')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(VisualizeStatesConfig, hydra_cfg)
    if not cfg.run:
        raise ValueError('Pass the ego run directory, e.g. run=runs/oc_cecp')
    run = RunDir.open(cfg.run)
    mi_dir = run.path / 'eval' / cfg.mi_eval_name
    mi_cfg = read_eval_config(mi_dir)
    shard_dir = mi_dir / cfg.layout / f'agent_{cfg.agent_id}'
    if not (shard_dir / 'raw_rows.npz').exists():
        raise FileNotFoundError(f'{shard_dir}/raw_rows.npz not found; run evaluate_partner_stage_mi.py with save_raw_rows=true.')
    summary = json.loads((shard_dir / 'summary.json').read_text())
    n_eps, length = int(mi_cfg['episodes_per_partner']), int(mi_cfg['episode_length'])
    selected = select_states(shard_dir, cfg, n_eps)

    # Rebuild the collection setup: same partner pool and order, env, start state and shard seed.
    pool, _ = resolve_partner_pool(PartnerPoolConfig(**mi_cfg['partners']), ego_run=run)
    recorded = [r['partner_name'] for r in csv.DictReader(open(mi_dir / 'partners.csv'))]
    if [p.name for p in pool] != recorded:
        raise ValueError(f'The partner pool resolves differently now than in {mi_dir}/partners.csv; the replay would not match.')
    loader = PolicyLoader()
    partners = [loader.load_entry(p.entry, run) for p in pool]
    ego = loader.load(run, cfg.agent_id)
    ev = make_eval_env(run, mi_cfg['env_overrides'])
    eval_env = FrozenAssignmentEnv(ev.env, n_eps, env_agent_to_role=[[0, 1]] * n_eps)
    env_const = eval_env_const(eval_env, ev.env_mode, False)
    start = tile_initial_state(*fixed_initial_state(ev.env, cfg.layout), n_eps)
    check_tiled_state(*start, eval_env, env_const)
    collect = aot_compile(make_fixed_start_ego_collector(eval_env, length, env_const, ego.model_fn, partners[0].model_fn,
                                                         replay_output(hash_multipliers(int(flatten_obs(start[1][0]).shape[-1])))),
                          jax.random.key(0), ego.model_state, partners[0].model_state, *start, name='state replayer')

    recovered = {}
    rng = jax.random.key(int(summary['shard_seed']))
    wanted = {s['partner_idx'] for s in selected}
    for partner_idx, partner in enumerate(partners):
        rng, rng_call = jax.random.split(rng)   # one split per partner, replayed or not
        if partner_idx not in wanted:
            continue
        (hashes, obs, states), _ = collect(rng_call, ego.model_state, partner.model_state, *start)
        hashes, obs = np.asarray(hashes), np.asarray(obs)
        for s in (s for s in selected if s['partner_idx'] == partner_idx):
            t, e = s['timestep'], s['env_idx']
            if not (np.array_equal(hashes[t, e], s['state_hash']) and np.array_equal(hash_obs_numpy(obs[t, e]), s['state_hash'])):
                raise AssertionError(f'State {s["state_id"]}: the replay did not reproduce the collected rollout '
                                     f'(partner {partner_idx}, episode {s["episode_id"]}, t={t}).')
            state = jax.tree_util.tree_map(lambda x: np.asarray(x[t, e]), states)
            raw = state._state
            got = (int(raw.agent_pos[0, 1]), int(raw.agent_pos[0, 0]), int(raw.agent_inv[0]), int(raw.agent_dir_idx[0]))
            if got != s['ego']:
                raise AssertionError(f'State {s["state_id"]}: replayed ego (row, col, inv, dir) {got} != stored {s["ego"]}.')
            recovered[s['state_id']] = state
    print(f'Recovered and verified {len(recovered)} state(s).')

    eval_name = cfg.eval_name or f'partner_stage_mi_states/{cfg.layout}/agent_{cfg.agent_id}'
    out_dir = eval_output_dir(run, eval_name, asdict(cfg), cfg.overwrite)
    names = stage_names(3)
    header = (f'{run_label(run.name)} agent {cfg.agent_id} on {OC_ORIGINAL_LAYOUT_NAMES.get(cfg.layout, cfg.layout)}   '
              f'I(a;z|s)={summary["mi_uniform_true"]:.4f} (uniform) over {summary["n_states_qualifying"]} qualifying states')
    frames = {sid: render_grid_frame(ev.env, state_for_rendering(state, red_seat=0)) for sid, state in recovered.items()}
    csv_rows = []
    for s in selected:
        state = recovered[s['state_id']]
        plot_state(out_dir / f'state_rank{s["rank"]:02d}_id{s["state_id"]}', s, frames[s['state_id']], state, names,
                   pool[s['partner_idx']].name, header)
        raw = state._state
        csv_rows.append({
            'rank': s['rank'], 'state_id': s['state_id'], 'delta_h': s['delta_h'], 'weight': s['weight'],
            'ego_row': s['ego'][0], 'ego_col': s['ego'][1], 'ego_inv': INVENTORY_NAMES.get(s['ego'][2], s['ego'][2]),
            'ego_dir': ORIENTATION_NAMES[s['ego'][3]], 'partner_row': int(raw.agent_pos[1, 1]), 'partner_col': int(raw.agent_pos[1, 0]),
            'partner_inv': INVENTORY_NAMES.get(int(raw.agent_inv[1]), int(raw.agent_inv[1])),
            'partner_dir': ORIENTATION_NAMES[int(raw.agent_dir_idx[1])], 'first_timestep': s['first_timestep'],
            'replay_partner': pool[s['partner_idx']].name, 'replay_episode_id': s['episode_id'], 'replay_timestep': s['timestep'],
            **{f'n_{n}': int(s['num_samples'][k]) for k, n in enumerate(names)},
            **{f'argmax_{n}': ACTION_NAMES[int(np.argmax(s['mean_probs'][k]))] for k, n in enumerate(names)},
            **{f'p_{n}_{a}': float(s['mean_probs'][k, i]) for k, n in enumerate(names) for i, a in enumerate(ACTION_NAMES[:s['mean_probs'].shape[1]])},
        })
    write_csv(out_dir / 'top_states.csv', csv_rows)

    n_cols = min(5, len(selected))
    fig, axes = plt.subplots(int(np.ceil(len(selected) / n_cols)), n_cols, figsize=(2.6 * n_cols, 2.9 * np.ceil(len(selected) / n_cols)), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, s in zip(axes.ravel(), selected):
        ax.set_axis_on()
        draw_frame(ax, frames[s['state_id']], recovered[s['state_id']])
        ax.set_xticks([])
        ax.set_yticks([])
        top = '/'.join(ACTION_NAMES[int(np.argmax(s['mean_probs'][k]))] for k in range(3))
        ax.set_title(f'#{s["rank"]} id {s["state_id"]}  JSD={s["delta_h"]:.3f}\nargmax e/m/l: {top}', fontsize=7)
    fig.suptitle(f'{header}\nTop-{len(selected)} states by per-state JSD; ego = red, partner = blue', fontsize=9, y=1.0)
    fig.tight_layout()
    save_fig(fig, out_dir / 'top_states')
    print(f'Wrote {out_dir}')


if __name__ == '__main__':
    main()
