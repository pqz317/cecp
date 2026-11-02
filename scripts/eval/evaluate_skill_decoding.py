"""Decode a partner's skill level (training stage) from a cooperator's latent state (paper Fig 1c,
10a, 10b).

Partners are training checkpoints of a CEC population; the step checkpoints selected for each partner
agent are labelled early / mid / late (see `cecp.eval.common.partner_pool`). For each ego agent the
script rolls out `num_episodes_per_partner` episodes with every partner, records the ego's latent at
every step, and reports how well a cross-validated LDA decodes the partner's stage from the latent at
each timestep. 'best' partner checkpoints are played (they shift the RNG streams of later partners,
as in the paper runs) but carry no stage label.

Writes to <run>/eval/<eval_name>/agent_<i>/:
  checkpoint_stage_decoding_accuracy_by_timestep.csv   timestep, accuracy_all_latent_dims, fold_<k>_accuracy
  partners.csv, per_episode_metrics.csv, summary.json
  latents_by_run.npy, pc_axes.npy, pc_scores_traj.npy, explained_variance_ratio.npy, stage_mask.npy,
  stage_label_ids.npy
  checkpoint_stage_decoding_accuracy.png, pca_trajectories_early_late_mean_subsampled.png (Fig 1c/10b)

Paper Fig 10a:
    python scripts/eval/evaluate_skill_decoding.py run=runs/oc_cecp             # partners from partners.yaml
    python scripts/eval/evaluate_skill_decoding.py run=runs/oc_cec latent_key=rnn_hidden \
        partners.num_checkpoints_per_agent=3                                     # partners from its own population
    python scripts/plot/plot_skill_decoding.py runs/oc_cecp runs/oc_cec
"""
import functools
import json
from dataclasses import asdict, dataclass, field
from math import ceil, sqrt

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PartnerPoolConfig, PolicyLoader, config_from_hydra, eval_env_const, eval_output_dir, make_eval_env,
    model_signature, resolve_partner_pool, stage_names, write_csv,
)
from cecp.eval.plotting import save_fig
from cecp.eval.rollouts import EgoStep, aot_compile, make_ego_partner_collector
from cecp.policy.actor_critic import ActorWrapperAgent
from cecp.runs import RunDir

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class SkillDecodingConfig:
    run: str = ''                           # the ego run (CECP cooperators or a CEC population)
    agent_ids: list[int] = field(default_factory=lambda: [0])
    latent_key: str = 'readout'             # 'readout' (256-d readout layer) or 'rnn_hidden' (flattened LSTM state)
    partners: dict = field(default_factory=dict)   # PartnerPoolConfig fields
    eval_name: str = 'skill_decoding'
    overwrite: bool = False
    num_episodes_per_partner: int = 20
    episode_length: int = 100
    seed: int | None = None                 # None -> the run's general.seed
    eval_seed_offset: int = int(3e8)
    env_overrides: dict = field(default_factory=dict)
    ego_role: int = 0
    num_pcs: int = 10
    decode_ridge: float = 1e-4
    pca_traj_subsample_n: int = 30
    save_plots: bool = True


def latent_output(latent_key: str):
    def output(s: EgoStep) -> jax.Array:
        if latent_key == 'rnn_hidden':
            return jnp.concatenate(jax.tree_util.tree_leaves(s.ego_next_agent_state['rnn_state']['rnn_carry']), axis=-1)
        return s.ego_info[latent_key]
    return output


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def kfold_lda_accuracy(X: jax.Array, y: jax.Array, K: int, ridge: float, n_folds: int = 5) -> jax.Array:
    """K-fold cross-validated LDA accuracy at every timestep. X: (T, N, d), y: (N,) -> (T, n_folds).

    Shared ridge-regularised covariance and log class priors; fold f holds out samples n % n_folds == f.
    """
    _, N, d = X.shape
    fold_ids = jnp.arange(N) % n_folds
    y_onehot = jax.nn.one_hot(y, K, dtype=jnp.float32)

    def accuracy_at(X_t):
        def fold_accuracy(fold):
            test_mask = (fold_ids == fold).astype(jnp.float32)
            train_mask = 1.0 - test_mask
            n_train = train_mask.sum()
            train_onehot = y_onehot * train_mask[:, None]
            class_counts = train_onehot.sum(0)
            class_means = (train_onehot.T @ X_t) / jnp.maximum(class_counts[:, None], 1e-12)
            residuals = (X_t - class_means[y]) * train_mask[:, None]
            dof = jnp.maximum(n_train - K, 1.0)
            sigma_inv = jnp.linalg.inv(residuals.T @ residuals / dof + ridge * jnp.eye(d, dtype=jnp.float32))
            sigma_inv_means = sigma_inv @ class_means.T
            quad = jnp.einsum('kd,dk->k', class_means, sigma_inv_means)
            log_priors = jnp.log(jnp.maximum(class_counts / jnp.maximum(n_train, 1e-12), 1e-12))
            scores = X_t @ sigma_inv_means - 0.5 * quad[None, :] + log_priors[None, :]
            correct = (jnp.argmax(scores, axis=1) == y).astype(jnp.float32)
            return (correct * test_mask).sum() / jnp.maximum(test_mask.sum(), 1e-12)
        return jax.vmap(fold_accuracy)(jnp.arange(n_folds))

    return jax.vmap(accuracy_at)(X)


# -- plots ---------------------------------------------------------------------------------------

def _pc_pair_axes(num_pcs: int):
    n_pairs = min(max(num_pcs - 1, 0), 9)
    n_cols = int(ceil(sqrt(max(n_pairs, 1))))
    n_rows = int(ceil(max(n_pairs, 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True, squeeze=False)
    return fig, axes.ravel(), n_pairs


def plot_pca_mean_with_subsample(traj: np.ndarray, labels: np.ndarray, names: list[str], out_stem, subsample_n: int) -> None:
    """Subsampled single-episode PC trajectories (thin) and each class's mean trajectory (bold)."""
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], size=min(subsample_n, int((labels == c).sum())), replace=False)
                          for c in np.unique(labels)])
    colors = [plt.get_cmap('tab10')(i) for i in range(len(names))]
    means = [traj[:, labels == c, :].mean(axis=1) if (labels == c).any() else None for c in range(len(names))]
    fig, axes, n_pairs = _pc_pair_axes(traj.shape[2])
    for p in range(n_pairs):
        ax = axes[p]
        for run_idx in idx:
            ax.plot(traj[:, run_idx, p], traj[:, run_idx, p + 1], color=colors[labels[run_idx]], alpha=0.3, linewidth=1.0)
        for c, mean in enumerate(means):
            if mean is not None:
                ax.plot(mean[:, p], mean[:, p + 1], color=colors[c], linewidth=4.0)
                ax.scatter(mean[0, p], mean[0, p + 1], color='black', s=70, zorder=5)
        ax.set(xlabel=f'PC{p}', ylabel=f'PC{p + 1}', title=f'PC{p} vs PC{p + 1}')
        ax.grid(alpha=0.25)
    for ax in axes[n_pairs:]:
        ax.axis('off')
    axes[0].legend(handles=[plt.Line2D([0], [0], color=colors[i], lw=2, label=n) for i, n in enumerate(names)], loc='best')
    save_fig(fig, out_stem, dpi=150)


def plot_decoding(acc: np.ndarray, fold_accs: np.ndarray, chance: float, out_stem) -> None:
    t = np.arange(len(acc))
    std = fold_accs.std(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.fill_between(t, acc - std, acc + std, color='black', alpha=0.15)
    ax.plot(t, acc, color='black', linewidth=2.0, label='Overall')
    ax.axhline(chance, linestyle='--', color='gray', linewidth=1.0, label=f'Chance ({chance:.3f})')
    ax.set(xlim=(0, len(t) - 1), ylim=(0.0, 1.0), xlabel='Timestep in episode', ylabel='Decoding accuracy',
           title='Decoding of partner checkpoint stage')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    save_fig(fig, out_stem, dpi=150)


# -- evaluation ----------------------------------------------------------------------------------

def evaluate_agent(cfg: SkillDecodingConfig, run: RunDir, agent_id: int, loader: PolicyLoader, pool, partners,
                   eval_env, env_const, out_dir, compiled: dict) -> dict:
    ego = loader.load(run, agent_id, 'best')
    seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed
    rng_eval = jax.random.key(seed + cfg.eval_seed_offset)
    ego_sig = model_signature(ego.model_fn, ego.model_state)

    latents, partner_rows, episode_rows = [], [], []
    for partner_idx, (pool_partner, partner) in enumerate(tqdm(list(zip(pool, partners)), desc=f'agent {agent_id}')):
        key = (ego_sig, model_signature(partner.model_fn, partner.model_state))
        if key not in compiled:
            collect = make_ego_partner_collector(eval_env, cfg.episode_length, env_const, ego.model_fn,
                                                 ActorWrapperAgent(partner.model_fn), latent_output(cfg.latent_key), cfg.ego_role)
            compiled[key] = aot_compile(collect, jax.random.key(0), ego.model_state, partner.model_state, name='latent collector')
        # One split per partner, 'best' partners included (they consume a key even though they are unlabelled).
        rng_eval, rng_partner = jax.random.split(rng_eval)
        latents_seq, returns = compiled[key](rng_partner, ego.model_state, partner.model_state)
        latents_seq, returns = np.asarray(latents_seq), np.asarray(returns)
        step = None if pool_partner.entry.ckpt == 'best' else int(pool_partner.entry.ckpt)
        partner_rows.append({'partner_index': partner_idx, 'partner_name': pool_partner.name, 'run': pool_partner.entry.run,
                             'agent_id': pool_partner.entry.agent_id, 'checkpoint_step': step, 'stage': pool_partner.stage})
        for e in range(latents_seq.shape[1]):
            latents.append(latents_seq[:, e])
            episode_rows.append({'partner_index': partner_idx, 'partner_name': pool_partner.name, 'checkpoint_step': step,
                                 'episode_index_within_partner': e, 'coop_episode_return': float(returns[e])})
    latents_by_run = np.stack(latents)                                    # (N, T, d)
    n_runs, n_timesteps, n_units = latents_by_run.shape

    stage_per_run = [p.stage for p in pool for _ in range(cfg.num_episodes_per_partner)]
    stage_mask = np.array([s is not None for s in stage_per_run])
    y = np.array([s for s in stage_per_run if s is not None], dtype=int)
    names = list(stage_names(int(y.max()) + 1))

    # PCA over all (timestep, episode) latents, 'best' partners included.
    latent_matrix = np.transpose(latents_by_run, (2, 1, 0)).reshape(n_units, n_timesteps * n_runs)
    centered = latent_matrix - latent_matrix.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    num_pcs = min(max(1, cfg.num_pcs), n_units)
    pc_axes = U[:, :num_pcs]
    pc_scores_traj = (pc_axes.T @ centered).T.reshape(n_timesteps, n_runs, num_pcs)
    explained = S ** 2 / np.maximum(np.sum(S ** 2), 1e-12)

    features = np.transpose(latents_by_run, (1, 0, 2))[:, stage_mask, :]  # (T, N_staged, d)
    fold_accs = np.asarray(kfold_lda_accuracy(jnp.asarray(features, jnp.float32), jnp.asarray(y, jnp.int32),
                                              len(names), cfg.decode_ridge, 5), dtype=np.float64)
    acc = fold_accs.mean(axis=1)
    chance = 1.0 / len(names)

    agent_dir = out_dir / f'agent_{agent_id}'
    agent_dir.mkdir()
    for name, arr in [('latents_by_run', latents_by_run), ('pc_axes', pc_axes), ('pc_scores_traj', pc_scores_traj),
                      ('explained_variance_ratio', explained), ('stage_mask', stage_mask), ('stage_label_ids', y)]:
        np.save(agent_dir / f'{name}.npy', arr)
    write_csv(agent_dir / 'partners.csv', partner_rows)
    write_csv(agent_dir / 'per_episode_metrics.csv', episode_rows)
    write_csv(agent_dir / 'checkpoint_stage_decoding_accuracy_by_timestep.csv', [
        {'timestep': t, 'accuracy_all_latent_dims': float(acc[t]),
         **{f'fold_{f}_accuracy': float(fold_accs[t, f]) for f in range(fold_accs.shape[1])}}
        for t in range(n_timesteps)])

    if cfg.save_plots:
        staged_traj = pc_scores_traj[:, stage_mask, :]
        plot_decoding(acc, fold_accs, chance, agent_dir / 'checkpoint_stage_decoding_accuracy')
        if 'early' in names and 'late' in names:
            keep = (y == names.index('early')) | (y == names.index('late'))
            plot_pca_mean_with_subsample(staged_traj[:, keep, :], np.where(y[keep] == names.index('early'), 0, 1),
                                         ['early', 'late'], agent_dir / 'pca_trajectories_early_late_mean_subsampled',
                                         cfg.pca_traj_subsample_n)

    summary = {
        'run': str(run.path), 'agent_id': agent_id, 'latent_key': cfg.latent_key, 'num_partners': len(pool),
        'num_episodes_per_partner': cfg.num_episodes_per_partner, 'episode_length': cfg.episode_length, 'seed': seed,
        'num_runs_total': n_runs, 'num_runs_with_stage': int(stage_mask.sum()), 'num_latent_units': n_units,
        'stage_class_names': names, 'stage_chance_level': chance,
        'mean_acc_stage_all_dims': float(acc.mean()), 'max_acc_stage_all_dims': float(acc.max()),
        'top_explained_variance_ratio': [float(x) for x in explained[:num_pcs]],
    }
    (agent_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'agent {agent_id}: mean stage decoding accuracy {acc.mean():.4f} (chance {chance:.4f}) -> {agent_dir}')
    return summary


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_skill_decoding')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(SkillDecodingConfig, hydra_cfg)
    if not cfg.run:
        raise ValueError('Pass the ego run directory, e.g. run=runs/oc_cecp')
    run = RunDir.open(cfg.run)
    pool, selection = resolve_partner_pool(PartnerPoolConfig(**cfg.partners), ego_run=run)
    if len({p.stage for p in pool if p.stage is not None}) < 2:
        raise ValueError(f'Skill decoding needs at least 2 partner stages; the partner selection {selection} gives fewer.')

    ev = make_eval_env(run, cfg.env_overrides, game_length=cfg.episode_length)
    if ev.num_players != 2:
        raise ValueError(f'Skill decoding needs 2-player games, got {ev.num_players} players.')
    n = cfg.num_episodes_per_partner
    eval_env = FrozenAssignmentEnv(ev.env, n, env_agent_to_role=[[0, 1]] * n)
    env_const = eval_env_const(eval_env, ev.env_mode, ev.eval_on_reserved)

    out_dir = eval_output_dir(run, cfg.eval_name, asdict(cfg) | {'partner_selection': selection}, cfg.overwrite)
    loader = PolicyLoader()
    partners = [loader.load_entry(p.entry, run) for p in pool]
    print(f'{len(pool)} partners ({sum(p.stage is not None for p in pool)} with a stage label) from {selection["runs"]}')
    compiled: dict = {}
    for agent_id in cfg.agent_ids:
        evaluate_agent(cfg, run, int(agent_id), loader, pool, partners, eval_env, env_const, out_dir, compiled)


if __name__ == '__main__':
    main()
