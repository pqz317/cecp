"""Per-member action distributions of a population at one grid cell, given what the acting agent holds
(paper Fig 5, 13; Overcooked).

Collects each member's self-play on one original layout (from its fixed start state), replays every
trajectory through every member (each from its own zero carry, as in evaluate_population_diversity),
keeps the steps where the acting agent stands on `cell` holding `inventory`, and reports for every
member j and action a the mean probability p_j(a) = E_s[pi_j(a | s)] over those states, the population
mixture, the per-state Delta H, and the same broken down by the agent's facing direction.

Writes to <run>/eval/<eval_name>/<layout>/cell_r<row>c<col>_<inventory>/:
  action_probs_by_agent.csv, action_probs_by_orientation.csv, state_context.csv, selfplay_returns.csv,
  per_state.npz, summary.json, and the figures action_probs_by_agent_matrix.png,
  action_probs_by_orientation.png and cell_context.png.

    python scripts/eval/evaluate_cell_action_distributions.py run=runs/oc_cec
"""
import json
from dataclasses import asdict, dataclass, field

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cecp.envs.spaces import Observation
from cecp.eval.common import PolicyLoader, config_from_hydra, eval_output_dir, make_eval_env, write_csv
from cecp.eval.display import OC_ORIGINAL_LAYOUT_NAMES
from cecp.eval.diversity import NUM_SEATS, assert_entropy_gap_bounds, entropy_gap_from_probs
from cecp.eval.overcooked import (
    ACTION_NAMES, CH_ONIONS_IN_POT, CH_POT_COOK_TIME, CH_POT_LOC, CH_SOUP_READY, GRID_SIZE, INVENTORY_ITEMS,
    ORIENTATION_NAMES, ORIGINAL_LAYOUTS, assert_action_semantics, describe_cell, fixed_initial_state, layout_grid,
    pot_cells, render_grid_frame, state_with_single_agent_at,
)
from cecp.eval.plotting import save_fig
from cecp.eval.rollouts import aot_compile, make_population_replay, make_selfplay_collector
from cecp.runs import RunDir

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'float32')   # see evaluate_population_diversity
WARN_REPLAY_VS_ONLINE = 1e-3
# Facing direction of the agent drawn on the cell in cell_context.png (the pot / counter it is about).
DISPLAY_ORIENTATION_BY_LAYOUT = {'counter_circuit_9': 'left', 'coord_ring_9': 'up'}


@dataclass
class CellActionConfig:
    run: str = ''                           # the population (a CEC run)
    eval_name: str = 'cell_action_distributions'
    overwrite: bool = False
    layout: str = 'coord_ring_9'
    cells: list[list[int]] = field(default_factory=lambda: [[3, 1]])   # [[row, col], ...]
    inventory: str = 'onion'                # empty | onion | plate | dish
    max_agents: int | None = 8
    episodes_per_source: int = 32
    episode_length: int | None = None       # None -> the run's general.game_length
    replay_batch_size: int = 16             # must divide 2 * episodes_per_source
    seed: int | None = None
    eval_seed_offset: int = int(4e8)
    min_states_per_orientation: int = 20    # orientations with fewer matched states are not plotted
    display_orientation: str | None = None  # None -> DISPLAY_ORIENTATION_BY_LAYOUT, else 'up'
    env_overrides: dict = field(default_factory=dict)


def distribution_stats(probs: np.ndarray) -> dict:
    """Per-member mean action distributions and Delta H from [N, M, A] probabilities over M states."""
    p = np.asarray(probs, dtype=np.float64)
    n_states = p.shape[1]
    mean_by_agent = p.mean(axis=1)
    stats = entropy_gap_from_probs(p[:, :, None, :])   # per state, then averaged
    return {
        'mean_by_agent': mean_by_agent,
        'sem_by_agent': p.std(axis=1, ddof=1) / np.sqrt(n_states) if n_states > 1 else np.zeros_like(mean_by_agent),
        'p_bar': mean_by_agent.mean(axis=0), 'n_states': n_states,
        'delta_h_mean': float(stats['delta_h'].mean()),
        'delta_h_sem': float(stats['delta_h'].std(ddof=1) / np.sqrt(n_states)) if n_states > 1 else 0.0,
        'h_bar_mean': float(stats['h_bar'].mean()), 'h_mean': float(stats['h_mean'].mean()),
        'delta_h_per_state': stats['delta_h'][:, 0], 'h_bar_per_state': stats['h_bar'][:, 0],
    }


def agent_rows(agent_ids, stats: dict, extra: dict | None = None) -> list[dict]:
    rows = []
    for label, probs, sems in [*[(int(a), stats['mean_by_agent'][i], stats['sem_by_agent'][i]) for i, a in enumerate(agent_ids)],
                               ('population_mean', stats['p_bar'], np.full(len(ACTION_NAMES), np.nan))]:
        rows.append({**(extra or {}), 'agent_id': label, 'n_states': int(stats['n_states']),
                     **{f'p_{n}': float(probs[a]) for a, n in enumerate(ACTION_NAMES)},
                     **{f'sem_{n}': float(sems[a]) for a, n in enumerate(ACTION_NAMES)},
                     'top_action': ACTION_NAMES[int(np.argmax(probs))], 'top_action_prob': float(probs.max())})
    return rows


# -- plots ---------------------------------------------------------------------------------------

def _annotated_matrix(ax, matrix: np.ndarray, col_labels, title: str):
    """Actions x members heatmap of mean action probabilities, values printed in the cells."""
    vmax = float(min(1.0, max(0.4, np.ceil(np.nanmax(matrix) * 10.0) / 10.0)))
    im = ax.imshow(matrix, cmap='viridis', vmin=0.0, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_yticks(np.arange(len(ACTION_NAMES)), ACTION_NAMES, fontsize=14)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, fontsize=11)
    ax.set_xlabel('agent ID', fontsize=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if matrix[i, j] < 0.55 * vmax else 'black')
    ax.axvline(matrix.shape[1] - 1.5, color='white', linewidth=2.0)   # the mixture column summarises the others
    ax.set_title(title, fontsize=12)
    return im


def _colorbar(fig, im, ax, **kwargs) -> None:
    cbar = fig.colorbar(im, ax=ax, **kwargs)
    cbar.set_label(r'$E_s[\pi_j(a|s)]$', fontsize=14)


def plot_agent_matrix(out_dir, stats: dict, agent_ids, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(0.65 * (len(agent_ids) + 1) + 2.8, 4.6))
    matrix = np.vstack([stats['mean_by_agent'], stats['p_bar'][None]]).T
    im = _annotated_matrix(ax, matrix, [str(a) for a in agent_ids] + [r'$\bar{\pi}$'], 'Mean action probability')
    _colorbar(fig, im, ax, shrink=0.8, pad=0.03)
    fig.suptitle(subtitle, fontsize=10.5, y=1.005)
    save_fig(fig, out_dir / 'action_probs_by_agent_matrix')


def plot_orientation_panels(out_dir, by_orientation: dict[int, dict], agent_ids, subtitle: str) -> None:
    if not by_orientation:
        return
    fig, axes = plt.subplots(len(by_orientation), 1, figsize=(0.65 * (len(agent_ids) + 1) + 2.8, 3.9 * len(by_orientation)),
                             squeeze=False, layout='constrained')
    for ax, (orientation, stats) in zip(axes[:, 0], sorted(by_orientation.items())):
        matrix = np.vstack([stats['mean_by_agent'], stats['p_bar'][None]]).T
        im = _annotated_matrix(ax, matrix, [str(a) for a in agent_ids] + [r'$\bar{\pi}$'],
                               f'facing {ORIENTATION_NAMES[orientation]}: n={stats["n_states"]}, $\\Delta H$={stats["delta_h_mean"]:.3f}')
        _colorbar(fig, im, ax, shrink=0.9, pad=0.02)
    fig.suptitle(f'Action distribution by facing direction\n{subtitle}', fontsize=12)
    save_fig(fig, out_dir / 'action_probs_by_orientation')


def plot_cell_context(out_dir, layout: str, cell, background: np.ndarray, timesteps: np.ndarray, subtitle: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].imshow(background, extent=(-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5), interpolation='nearest')
    axes[0].add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, fill=False, edgecolor='#e03131', linewidth=2.5, zorder=3))
    axes[0].set(xticks=np.arange(GRID_SIZE), yticks=np.arange(GRID_SIZE), xlabel='x (col)', ylabel='y (row)')
    axes[0].set_title(f'{OC_ORIGINAL_LAYOUT_NAMES.get(layout, layout)}: cell {tuple(cell)}', fontsize=10)
    axes[1].hist(timesteps, bins=32, color='#1f4e79', alpha=0.85)
    axes[1].set(xlabel='Timestep', ylabel='Matched states')
    axes[1].set_title('When the matched states occur', fontsize=10)
    fig.suptitle(subtitle, fontsize=9, y=1.02)
    save_fig(fig, out_dir / 'cell_context')


# -- evaluation ----------------------------------------------------------------------------------

def collect_selected_states(cfg: CellActionConfig, ev, population, collect, replay, stacked, start, cell, inventory_index,
                            pots, rng):
    """Self-play every member, replay through all members, keep the steps matching (cell, inventory)."""
    n_seq, length, bs = 2 * cfg.episodes_per_source, ev.game_length, cfg.replay_batch_size
    agent_ids = [p.agent_id for p in population]
    chunks = {k: [] for k in ('probs', 'source_agent', 'seat', 'timestep', 'orientation', 'pot_onions', 'pot_busy')}
    return_rows, max_err = [], 0.0
    for policy in tqdm(population, desc='self-play sources', leave=False):
        rng, rng_collect = jax.random.split(rng)
        obs, online, pos, inv, direction, done, returns = map(np.asarray, collect(rng_collect, policy.model_state, *start))
        if done[:, :, :-1].any():
            raise AssertionError('An episode ended before the last recorded step.')
        obs = obs.reshape(n_seq, length, *obs.shape[3:])
        pos, inv, direction = pos.reshape(n_seq, length, 2), inv.reshape(n_seq, length), direction.reshape(n_seq, length)
        probs = np.concatenate([np.asarray(replay(jax.random.key(0), stacked, Observation({'grid_2d': jnp.asarray(obs[s:s + bs])})))
                                for s in range(0, n_seq, bs)], axis=1)   # [N, n_seq, T, A]
        err = float(np.max(np.abs(probs[agent_ids.index(policy.agent_id)] - online.reshape(n_seq, length, -1))))
        max_err = max(max_err, err)
        if err > WARN_REPLAY_VS_ONLINE:
            print(f'WARNING: agent {policy.agent_id}: replay differs from the online forward by {err:.3e}.')
        seq, t = np.nonzero((pos[..., 0] == cell[0]) & (pos[..., 1] == cell[1]) & (inv == inventory_index))
        if seq.size:
            matched_obs = obs[seq, t]
            chunks['probs'].append(probs[:, seq, t, :])
            chunks['source_agent'].append(np.full(seq.size, policy.agent_id, np.int32))
            chunks['seat'].append((seq % NUM_SEATS).astype(np.int32))
            chunks['timestep'].append(t.astype(np.int32))
            chunks['orientation'].append(direction[seq, t].astype(np.int32))
            chunks['pot_onions'].append(np.stack([matched_obs[:, r, c, CH_ONIONS_IN_POT] for r, c in pots], axis=1).astype(np.int32))
            chunks['pot_busy'].append(np.stack([(matched_obs[:, r, c, CH_POT_COOK_TIME] > 0) | (matched_obs[:, r, c, CH_SOUP_READY] > 0)
                                                for r, c in pots], axis=1))
            for r, c in pots:
                if obs[:, :, r, c, CH_POT_LOC].max() < 1:
                    raise AssertionError(f'Observation channel {CH_POT_LOC} shows no pot at {(r, c)}.')
        team = returns.mean(axis=1)
        return_rows.append({'source_agent_id': policy.agent_id, 'mean_return': float(team.mean()), 'std_return': float(team.std()),
                            'num_episodes': int(returns.shape[0]), 'matched_states': int(seq.size)})
    if not chunks['probs']:
        raise ValueError(f'No state matched cell {tuple(cell)} with inventory={cfg.inventory}; '
                         'check the cell against the printed layout grid or raise episodes_per_source.')
    selected = {k: np.concatenate(v, axis=1 if k == 'probs' else 0) for k, v in chunks.items()}
    return selected, return_rows, {'max_replay_vs_online_prob_error': max_err, 'n_scored_steps': len(population) * n_seq * length}, rng


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_cell_action_distributions')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(CellActionConfig, hydra_cfg)
    if not cfg.run:
        raise ValueError('Pass the population run directory, e.g. run=runs/oc_cec')
    if cfg.layout not in ORIGINAL_LAYOUTS:
        raise ValueError(f'layout must be one of {list(ORIGINAL_LAYOUTS)}, got {cfg.layout!r}.')
    if cfg.inventory not in INVENTORY_ITEMS:
        raise ValueError(f'inventory must be one of {sorted(INVENTORY_ITEMS)}, got {cfg.inventory!r}.')
    if (2 * cfg.episodes_per_source) % cfg.replay_batch_size:
        raise ValueError(f'replay_batch_size={cfg.replay_batch_size} must divide 2 * episodes_per_source.')
    run = RunDir.open(cfg.run)
    ev = make_eval_env(run, cfg.env_overrides, cfg.episode_length)
    if ev.env_mode != 'overcooked':
        raise ValueError(f'evaluate_cell_action_distributions supports Overcooked only, got {ev.env_mode}.')
    assert_action_semantics(ev.env, cfg.layout)
    print(f'{cfg.layout} (P plate pile, O onion pile, T pot, G serving, A agent start):')
    for i, row in enumerate(layout_grid(cfg.layout)):
        print(f'  {i}  ' + ' '.join(row))

    out_root = eval_output_dir(run, cfg.eval_name, asdict(cfg), cfg.overwrite)
    population = PolicyLoader().load_population(run, max_agents=cfg.max_agents)
    agent_ids = [p.agent_id for p in population]
    stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[p.model_state for p in population])
    start = fixed_initial_state(ev.env, cfg.layout)
    collect = aot_compile(make_selfplay_collector(ev.env, ev.game_length, cfg.episodes_per_source, population[0].model_fn),
                          jax.random.key(0), population[0].model_state, *start, name='self-play collector')
    example_obs = Observation({'grid_2d': jnp.zeros((cfg.replay_batch_size, ev.game_length, *start[1][0]['grid_2d'].shape[1:]),
                                                    start[1][0]['grid_2d'].dtype)})
    replay = aot_compile(make_population_replay(population[0].model_fn), jax.random.key(0), stacked, example_obs, name='population replay')
    pots = pot_cells(cfg.layout)
    inventory_index = INVENTORY_ITEMS[cfg.inventory]
    orientation_name = cfg.display_orientation or DISPLAY_ORIENTATION_BY_LAYOUT.get(cfg.layout, 'up')
    seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed
    rng = jax.random.key(seed + cfg.eval_seed_offset)

    for cell in cfg.cells:
        # One collection pass per cell (the RNG chain continues), so the cells' state sets are independent.
        selected, return_rows, validation, rng = collect_selected_states(
            cfg, ev, population, collect, replay, stacked, start, cell, inventory_index, pots, rng)
        out_dir = out_root / cfg.layout / f'cell_r{cell[0]}c{cell[1]}_{cfg.inventory}'
        out_dir.mkdir(parents=True)
        stats = distribution_stats(selected['probs'])
        assert_entropy_gap_bounds({'delta_h': stats['delta_h_per_state'], 'h_bar': stats['h_bar_per_state']},
                                  len(ACTION_NAMES), f'{cfg.layout} cell {cell}')
        subtitle = (f'{OC_ORIGINAL_LAYOUT_NAMES[cfg.layout]}, cell (row {cell[0]}, col {cell[1]}) '
                    f'[{describe_cell(cfg.layout, *cell)}], holding {cfg.inventory}; '
                    f'{stats["n_states"]} states, $\\Delta H$ = {stats["delta_h_mean"]:.3f} nats')

        write_csv(out_dir / 'action_probs_by_agent.csv', agent_rows(agent_ids, stats))
        orientation_rows, by_orientation = [], {}
        for o, name in enumerate(ORIENTATION_NAMES):
            mask = selected['orientation'] == o
            if mask.any():
                sub = distribution_stats(selected['probs'][:, mask, :])
                orientation_rows += agent_rows(agent_ids, sub, {'orientation': name, 'delta_h_mean_nats': sub['delta_h_mean']})
                if mask.sum() >= cfg.min_states_per_orientation:
                    by_orientation[o] = sub
        write_csv(out_dir / 'action_probs_by_orientation.csv', orientation_rows)
        context_rows = [{'variable': 'orientation', 'value': name, 'count': int((selected['orientation'] == o).sum())}
                        for o, name in enumerate(ORIENTATION_NAMES)]
        for k, pot in enumerate(pots):
            context_rows += [{'variable': f'pot{k}@{pot}_onions', 'value': str(n),
                              'count': int(((selected['pot_onions'][:, k] == n) & ~selected['pot_busy'][:, k]).sum())} for n in range(4)]
            context_rows.append({'variable': f'pot{k}@{pot}_onions', 'value': 'cooking_or_ready', 'count': int(selected['pot_busy'][:, k].sum())})
        context_rows += [{'variable': 'seat', 'value': str(s), 'count': int((selected['seat'] == s).sum())} for s in range(NUM_SEATS)]
        write_csv(out_dir / 'state_context.csv', context_rows)
        write_csv(out_dir / 'selfplay_returns.csv', return_rows)
        np.savez_compressed(out_dir / 'per_state.npz', **{k: v for k, v in selected.items()},
                            agent_ids=np.asarray(agent_ids, np.int32), delta_h=stats['delta_h_per_state'])

        plot_agent_matrix(out_dir, stats, agent_ids, subtitle)
        plot_orientation_panels(out_dir, by_orientation, agent_ids, subtitle)
        display_state = state_with_single_agent_at(start[0], tuple(cell), ORIENTATION_NAMES.index(orientation_name), inventory_index)
        plot_cell_context(out_dir, cfg.layout, cell, render_grid_frame(ev.env, display_state), selected['timestep'], subtitle)

        summary = {'run': str(run.path), 'layout': cfg.layout, 'cell': list(cell), 'cell_description': describe_cell(cfg.layout, *cell),
                   'inventory': cfg.inventory, 'agent_ids': agent_ids, 'action_names': ACTION_NAMES, 'seed': seed,
                   'n_matched_states': int(stats['n_states']), 'episodes_per_source': cfg.episodes_per_source,
                   'episode_length': ev.game_length, 'delta_h_mean_nats': stats['delta_h_mean'],
                   'delta_h_sem_nats': stats['delta_h_sem'], 'h_bar_mean_nats': stats['h_bar_mean'], 'h_mean_nats': stats['h_mean'],
                   'population_mean_probs': dict(zip(ACTION_NAMES, map(float, stats['p_bar']))),
                   'top_action_by_agent': {str(a): ACTION_NAMES[int(np.argmax(stats['mean_by_agent'][i]))] for i, a in enumerate(agent_ids)},
                   'validation': validation}
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f'{cfg.layout} cell {tuple(cell)} holding {cfg.inventory}: {stats["n_states"]} states, '
              f'Delta H {stats["delta_h_mean"]:.4f} nats -> {out_dir}')


if __name__ == '__main__':
    main()
