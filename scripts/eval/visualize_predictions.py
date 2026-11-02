"""Example frames of a CECP cooperator's next-state prediction (paper Fig 17, Overcooked).

Plays one episode of the cooperator (seat 0) with a partner on a fixed layout and saves, for every
step, the state at t, the true state at t+1 and the cooperator's predicted observation at t+1.

Writes to <run>/eval/<eval_name>/: step_<t>.png/.svg and summary.json.

    python scripts/eval/visualize_predictions.py run=runs/oc_cecp      # partner: runs/oc_cec agent 1 (paper)
"""
import json
from dataclasses import asdict, dataclass, field

import hydra
import jax
import matplotlib
import numpy as np
from omegaconf import DictConfig

from cecp.controller.base import BaseController
from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PolicyLoader, config_from_hydra, env_section, eval_env_const, eval_output_dir, make_eval_env, resolve_run_path,
)
from cecp.eval.plotting import save_fig
from cecp.eval.rollouts import aot_compile
from cecp.policy.actor_critic import ActorCriticPPOAgent, ActorWrapperAgent
from cecp.runs import PARTNERS_FILE, RunDir
from cecp.utils import split_rng_to_list

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TILE = 32


@dataclass
class VisualizePredictionsConfig:
    run: str = ''                           # a CECP run trained with the predictive objective
    agent_id: int = 0
    partner_run: str = ''                   # '' -> the first partner run in the ego's partners.yaml (else the ego run)
    partner_agent_id: int = 1
    partner_ckpt: str = 'best'
    eval_name: str = 'prediction_frames'
    overwrite: bool = False
    num_steps: int = 50
    seed: int | None = None                 # None -> the run's general.seed
    eval_seed_offset: int = int(3e8)
    env_overrides: dict = field(default_factory=dict)
    agent_view_size: int = 5
    figure_dpi: int = 160


def compile_predictive_rollout(eval_env, num_steps, model_fn_i, model_fn_j, state_i, state_j, env_const):
    """`(rng, state_i, state_j) -> (env states, next env states, predicted next obs)`, each stacked over steps.
    Policy i is a PPO agent so that its extra data carries the predicted next observation."""
    agents = [ActorCriticPPOAgent(model_fn_i), ActorWrapperAgent(model_fn_j)]
    batch_sizes = eval_env.get_agent_batch_size()

    def rollout(rng, model_state_i, model_state_j):
        model_states = [model_state_i, model_state_j]
        rng, rng_init_lst = split_rng_to_list(rng, len(agents))
        agent_states = [a.init_agent_state(r, b) for r, a, b in zip(rng_init_lst, agents, batch_sizes)]
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = eval_env.reset(rng_reset, env_const)

        def step(carry, _):
            rng, agent_states, env_state, obs_lst = carry
            rng, agent_states, next_env_state, next_obs_lst, _, _, _, info = BaseController.rollout_step(
                env_fn=eval_env, agent_lst=agents, rng=rng, model_state_lst=model_states, agent_state_lst=agent_states,
                env_const=env_const, env_state=env_state, obs_lst=obs_lst, is_collect_extra_data=True)
            return (rng, agent_states, next_env_state, next_obs_lst), (env_state, next_env_state, info['agent-0-extra-data']['pred_next_obs'])

        _, outputs = jax.lax.scan(step, (rng, agent_states, env_state, obs_lst), None, length=num_steps)
        return outputs

    return aot_compile(rollout, jax.random.key(0), state_i, state_j, name='predictive rollout')


def save_step_figure(out_stem, frame_t, frame_tp1, pred_frame_tp1, t, dpi) -> None:
    # Crop the outer wall ring and the grey margin of the rendered frames; the predicted frame has no wall ring.
    mask = np.any(frame_t != frame_t[0, 0], axis=2)
    row_max, col_max = int(np.where(mask.any(axis=1))[0][-1]), int(np.where(mask.any(axis=0))[0][-1])
    frames = [frame_t[TILE:row_max + 1, TILE:col_max + 1], frame_tp1[TILE:row_max + 1, TILE:col_max + 1],
              pred_frame_tp1[:row_max + 1 - TILE, :col_max + 1 - TILE]]
    h, w = frames[0].shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(3 * w / h * 5, 5))
    for ax, frame, title in zip(axes, frames, [f'state_t (t={t})', f'true_state_t+1 (t={t + 1})', 'pred_state_t+1']):
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    save_fig(fig, out_stem, dpi=dpi)


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='visualize_predictions')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(VisualizePredictionsConfig, hydra_cfg)
    if not cfg.run:
        raise ValueError('Pass the CECP run directory, e.g. run=runs/oc_cecp')
    run = RunDir.open(cfg.run)
    if cfg.partner_run:
        partner_run = RunDir.open(cfg.partner_run)
    elif (run.path / PARTNERS_FILE).exists():
        partner_run = RunDir.open(resolve_run_path(run.read_partners().partner_runs[0], run))
    else:
        partner_run = run
    loader = PolicyLoader()
    ego = loader.load(run, cfg.agent_id, 'best')
    if not ego.model_fn.has_predictive_head():
        raise ValueError(f'{run.path} agent {cfg.agent_id} has no next-state prediction head.')
    partner = loader.load(partner_run, cfg.partner_agent_id, cfg.partner_ckpt if cfg.partner_ckpt == 'best' else int(cfg.partner_ckpt))

    ev = make_eval_env(run, cfg.env_overrides)
    if ev.env_mode != 'overcooked':
        raise ValueError(f'visualize_predictions supports Overcooked only, got {ev.env_mode}.')
    eval_env = FrozenAssignmentEnv(ev.env, 1, env_agent_to_role=[[0, 1]])
    env_const = eval_env_const(eval_env, ev.env_mode, ev.eval_on_reserved)
    rollout = compile_predictive_rollout(eval_env, cfg.num_steps, ego.model_fn, partner.model_fn,
                                         ego.model_state, partner.model_state, env_const)
    seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed
    _, rng_rollout = jax.random.split(jax.random.key(seed + cfg.eval_seed_offset))
    env_states, next_env_states, pred_next_obs = rollout(rng_rollout, ego.model_state, partner.model_state)

    out_dir = eval_output_dir(run, cfg.eval_name, asdict(cfg) | {'partner_run_resolved': str(partner_run.path)}, cfg.overwrite)
    render_frame = ev.env.create_frame_renderer()
    for t in range(cfg.num_steps):
        state_t = jax.tree_util.tree_map(lambda x: x[t, 0], env_states)
        state_tp1 = jax.tree_util.tree_map(lambda x: x[t, 0], next_env_states)
        pred_tp1 = jax.tree_util.tree_map(lambda x: x[t], pred_next_obs)
        save_step_figure(out_dir / f'step_{t:04d}',
                         render_frame(state_t, agent_view_size=cfg.agent_view_size).astype(np.uint8),
                         render_frame(state_tp1, agent_view_size=cfg.agent_view_size).astype(np.uint8),
                         np.asarray(ev.env._render_predicted_obs_frame(pred_tp1, tile_size=TILE), dtype=np.uint8),
                         t, cfg.figure_dpi)
    summary = {'run': str(run.path), 'agent_id': cfg.agent_id, 'partner_run': str(partner_run.path),
               'partner_agent_id': cfg.partner_agent_id, 'partner_ckpt': cfg.partner_ckpt, 'seed': seed,
               'num_steps': cfg.num_steps, 'env': env_section(run, cfg.env_overrides)}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Wrote {cfg.num_steps} frames to {out_dir}')


if __name__ == '__main__':
    main()
