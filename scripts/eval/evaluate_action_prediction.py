"""How well a CECP cooperator's next-state prediction anticipates its partner's movement (paper Fig 9a,
AI side; Overcooked).

At every step the cooperator predicts the next observation given its own sampled action. The
partner's predicted move is read off the partner-orientation channels: a change of orientation
between the current and the predicted observation is a move in that direction; no change counts as
"no-move". Ground truth is the partner's actual action, with stay and interact merged into "no-move"
(5 classes, chance 0.2).

Writes to <run>/eval/<eval_name>/: accuracy_by_timestep.csv, accuracy_by_timestep.npy,
per_partner_accuracy_by_timestep.npy, confusion_matrix.npy, partners.csv, per_episode_metrics.csv and
summary.json (overall_accuracy = the Fig 9a AI value).

    python scripts/eval/evaluate_action_prediction.py run=runs/oc_cecp     # partners: its partners.yaml runs
    python scripts/plot/plot_action_prediction.py runs/oc_cecp
"""
import json
from dataclasses import asdict, dataclass, field

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from cecp.envs.batched import FrozenAssignmentEnv
from cecp.eval.common import (
    PartnerPoolConfig, PolicyLoader, config_from_hydra, eval_env_const, eval_output_dir, make_eval_env,
    model_signature, resolve_partner_pool, write_csv,
)
from cecp.eval.overcooked import PARTNER_ORIENTATION_CHANNELS
from cecp.eval.rollouts import EgoStep, aot_compile, make_ego_partner_collector
from cecp.policy.actor_critic import ActorWrapperAgent
from cecp.runs import RunDir

ACTION_CLASSES = ['right', 'down', 'left', 'up', 'no-move']
NO_MOVE = 4


@dataclass
class ActionPredictionConfig:
    run: str = ''                           # a CECP run trained with the predictive objective
    agent_id: int = 1                       # the paper figure used cooperator 1
    partners: dict = field(default_factory=dict)   # PartnerPoolConfig fields
    eval_name: str = 'action_prediction'
    overwrite: bool = False
    num_episodes_per_partner: int = 50
    episode_length: int = 256
    seed: int | None = None                 # None -> the run's general.seed
    eval_seed_offset: int = int(5e8)
    env_overrides: dict = field(default_factory=dict)
    ego_role: int = 0


def make_prediction_output(ego_model_fn):
    def output(s: EgoStep):
        # Same forward key as the policy step, conditioned on the action the ego actually sampled.
        _, _, _, pred_next_obs = ego_model_fn.forward_with_prediction(
            s.rng_forward, s.ego_model_state, s.ego_agent_state, s.ego_obs, s.ego_action)
        return pred_next_obs['grid_2d'], s.ego_obs['grid_2d'], s.partner_action['all']
    return output


def predicted_partner_move(current_obs: np.ndarray, pred_next_obs: np.ndarray) -> np.ndarray:
    """Partner move implied by the predicted observation: the new orientation if it changed, else no-move."""
    current_dir = current_obs[..., PARTNER_ORIENTATION_CHANNELS].sum(axis=(-3, -2)).argmax(-1)
    pred_dir = pred_next_obs[..., PARTNER_ORIENTATION_CHANNELS].sum(axis=(-3, -2)).argmax(-1)
    return np.where(current_dir != pred_dir, pred_dir, NO_MOVE)


def accuracy_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """pred, true: (T, B) action classes."""
    correct = pred == true
    n = len(ACTION_CLASSES)
    per_class = np.full((pred.shape[0], n), np.nan)
    for c in range(n):
        mask = true == c
        valid = mask.sum(1) > 0
        per_class[valid, c] = (correct & mask).sum(1)[valid] / mask.sum(1)[valid]
    confusion = np.array([[int(((true == t) & (pred == p)).sum()) for p in range(n)] for t in range(n)], dtype=np.int64)
    return {'per_timestep': correct.mean(1), 'per_class_per_timestep': per_class, 'confusion_matrix': confusion,
            'overall_accuracy': float(correct.mean())}


@hydra.main(version_base=None, config_path='../../configs/eval', config_name='evaluate_action_prediction')
def main(hydra_cfg: DictConfig) -> None:
    cfg = config_from_hydra(ActionPredictionConfig, hydra_cfg)
    if not cfg.run:
        raise ValueError('Pass the CECP run directory, e.g. run=runs/oc_cecp')
    run = RunDir.open(cfg.run)
    pool, selection = resolve_partner_pool(PartnerPoolConfig(**cfg.partners), ego_run=run)
    ev = make_eval_env(run, cfg.env_overrides, game_length=cfg.episode_length)
    if ev.env_mode != 'overcooked':
        raise ValueError(f'evaluate_action_prediction supports Overcooked only, got {ev.env_mode}.')
    loader = PolicyLoader()
    ego = loader.load(run, cfg.agent_id, 'best')
    if not ego.model_fn.has_predictive_head():
        raise ValueError(f'{run.path} agent {cfg.agent_id} has no next-state prediction head.')
    out_dir = eval_output_dir(run, cfg.eval_name, asdict(cfg) | {'partner_selection': selection}, cfg.overwrite)

    n = cfg.num_episodes_per_partner
    eval_env = FrozenAssignmentEnv(ev.env, n, env_agent_to_role=[[0, 1]] * n)
    env_const = eval_env_const(eval_env, ev.env_mode, ev.eval_on_reserved)
    partners = [loader.load_entry(p.entry, run) for p in pool]

    # Partners with the same architecture are rolled out together (vmapped). As in the paper runs, all
    # partners of a group share one RNG key, i.e. the same environment resets.
    groups: dict[tuple, list[int]] = {}
    for i, partner in enumerate(partners):
        groups.setdefault(model_signature(partner.model_fn, partner.model_state), []).append(i)

    seed = int(run.config['general']['seed']) if cfg.seed is None else cfg.seed
    rng_eval = jax.random.key(seed + cfg.eval_seed_offset)
    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # partner index -> (pred, true, returns)
    for members in groups.values():
        first = partners[members[0]]
        collect = make_ego_partner_collector(eval_env, cfg.episode_length, env_const, ego.model_fn,
                                             ActorWrapperAgent(first.model_fn), make_prediction_output(ego.model_fn), cfg.ego_role)
        rng_eval, rng_group = jax.random.split(rng_eval)
        if len(members) > 1:
            stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[partners[i].model_state for i in members])
            fn = aot_compile(jax.vmap(collect, in_axes=(None, None, 0)), jax.random.key(0), ego.model_state, stacked,
                             name='vmapped prediction collector')
            (pred_obs, cur_obs, partner_actions), returns = fn(rng_group, ego.model_state, stacked)
        else:
            fn = aot_compile(collect, jax.random.key(0), ego.model_state, first.model_state, name='prediction collector')
            (pred_obs, cur_obs, partner_actions), returns = jax.tree_util.tree_map(
                lambda x: x[None], fn(rng_group, ego.model_state, first.model_state))
        pred_obs, cur_obs, partner_actions, returns = map(np.asarray, (pred_obs, cur_obs, partner_actions, returns))
        for k, i in enumerate(members):
            results[i] = (predicted_partner_move(cur_obs[k], pred_obs[k]), np.minimum(partner_actions[k], NO_MOVE), returns[k])

    order = [i for members in groups.values() for i in members]   # group order, as in the paper runs
    per_partner = {i: accuracy_metrics(results[i][0], results[i][1]) for i in order}
    overall = accuracy_metrics(np.concatenate([results[i][0] for i in order], axis=1),
                               np.concatenate([results[i][1] for i in order], axis=1))
    per_partner_acc = np.stack([per_partner[i]['per_timestep'] for i in order])

    write_csv(out_dir / 'accuracy_by_timestep.csv', [
        {'timestep': t, 'overall_accuracy': float(overall['per_timestep'][t]),
         **{f'accuracy_{name}': (None if np.isnan(v) else float(v))
            for name, v in zip(ACTION_CLASSES, overall['per_class_per_timestep'][t])}}
        for t in range(cfg.episode_length)])
    write_csv(out_dir / 'partners.csv', [
        {'partner_index': i, 'partner_name': pool[i].name, 'run': pool[i].entry.run, 'agent_id': pool[i].entry.agent_id,
         'ckpt': pool[i].entry.ckpt, 'overall_accuracy': per_partner[i]['overall_accuracy']} for i in sorted(order)])
    write_csv(out_dir / 'per_episode_metrics.csv', [
        {'partner_index': i, 'partner_name': pool[i].name, 'episode_index': e, 'coop_episode_return': float(r)}
        for i in order for e, r in enumerate(results[i][2])])
    np.save(out_dir / 'confusion_matrix.npy', overall['confusion_matrix'])
    np.save(out_dir / 'accuracy_by_timestep.npy', overall['per_timestep'])
    np.save(out_dir / 'per_partner_accuracy_by_timestep.npy', per_partner_acc)   # rows in group order (= partners.csv order
                                                                                  # when all partners share an architecture)
    summary = {
        'run': str(run.path), 'agent_id': cfg.agent_id, 'num_partners': len(pool), 'seed': seed,
        'num_episodes_per_partner': n, 'episode_length': cfg.episode_length, 'action_class_names': ACTION_CLASSES,
        'chance_level': 1.0 / len(ACTION_CLASSES), 'overall_accuracy': overall['overall_accuracy'],
        'mean_per_timestep_accuracy': float(overall['per_timestep'].mean()),
        'per_partner_accuracy': {pool[i].name: per_partner[i]['overall_accuracy'] for i in order},
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Overall partner-move prediction accuracy {overall["overall_accuracy"]:.4f} (chance 0.2) -> {out_dir}')


if __name__ == '__main__':
    main()
