"""Helpers shared by the evaluation scripts.

Conventions:
- Inputs are run directories (see `cecp.runs`), never tags or checkpoint globs.
- The environment is rebuilt from the evaluated run's own config.yaml, plus explicit eval overrides.
- Outputs go to <run>/eval/<eval_name>/, next to the resolved eval config (eval_config.yaml).
"""
import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.struct import PyTreeNode

from ..config import env_config_from_mapping, from_mapping
from ..envs.make import EnvMode, make_env
from ..partners import build_manifest
from ..runs import CONFIG_FILE, PARTNERS_FILE, Checkpoint, PartnerEntry, RunDir, _to_plain, build_model_fn
from .plotting import EVAL_DIR

EVAL_CONFIG_FILE = 'eval_config.yaml'

T = TypeVar('T')


# -- output directories --------------------------------------------------------------------------

def eval_output_dir(run: RunDir, eval_name: str, eval_config: Mapping[str, Any], overwrite: bool = False) -> Path:
    """Create <run>/eval/<eval_name>/ and write the resolved eval config into it.

    An existing non-empty output directory is an error unless `overwrite`, in which case it is
    cleared first so no stale files from an earlier evaluation remain.
    """
    out_dir = run.path / EVAL_DIR / eval_name
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f'{out_dir} already exists; pass overwrite=true to replace it, '
                                  f'or eval_name=<new name> to keep both.')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EVAL_CONFIG_FILE).write_text(yaml.safe_dump(_to_plain(eval_config), sort_keys=False))
    return out_dir


def config_from_hydra(cls: type[T], cfg: Any) -> T:
    """Build an eval config dataclass from a Hydra DictConfig (unknown keys are an error)."""
    from omegaconf import OmegaConf
    return from_mapping(cls, OmegaConf.to_container(cfg, resolve=True))


def read_eval_config(out_dir: str | Path) -> dict[str, Any]:
    with open(Path(out_dir) / EVAL_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def resolve_run_path(path: str | Path, relative_to: RunDir | None = None) -> Path:
    """Locate a run directory referenced by another run (e.g. a partner run in partners.yaml).

    Such paths are stored as given at training time, often relative to the working directory then.
    If `path` doesn't resolve from here, fall back to a sibling of `relative_to` with the same name.
    """
    path = Path(path)
    if (path / CONFIG_FILE).exists():
        return path
    if relative_to is not None:
        sibling = relative_to.path.parent / path.name
        if (sibling / CONFIG_FILE).exists():
            return sibling
    raise FileNotFoundError(f'Run directory {path} not found' + (f' (also tried next to {relative_to.path})' if relative_to else ''))


# -- environments --------------------------------------------------------------------------------

# Env config keys that only affect training (reward shaping); ignored when checking that runs share an env.
_TRAIN_ONLY_ENV_KEYS = ('reward_shaping_steps', 'train_shaped_reward_factor')


def env_section(run: RunDir, env_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The run's `env` config section with `env_overrides` applied."""
    section = dict(run.config['env'])
    if env_overrides:
        unknown = set(env_overrides) - set(section)
        if unknown:
            raise ValueError(f'env_overrides has keys {sorted(unknown)} that are not in the env config of {run.path}')
        section.update(env_overrides)
    return section


def check_same_env(runs: Sequence[RunDir], env_overrides: Mapping[str, Any] | None = None) -> None:
    """Raise if the runs would be evaluated on different environments."""
    def key(run: RunDir) -> dict[str, Any]:
        section = {k: v for k, v in env_section(run, env_overrides).items() if k not in _TRAIN_ONLY_ENV_KEYS}
        return {**section, 'num_players': run.config['general']['num_players']}

    ref = key(runs[0])
    for run in runs[1:]:
        other = key(run)
        diff = sorted(k for k in ref.keys() | other.keys() if ref.get(k) != other.get(k))
        if diff:
            raise ValueError(f'{runs[0].path} and {run.path} have different env settings for {diff}; '
                             'evaluate them separately or align them with env_overrides.')


@dataclass
class EvalEnv:
    env: Any
    env_mode: EnvMode
    eval_on_reserved: bool      # evaluate on the held-out goal positions / layouts
    num_players: int
    game_length: int


def make_eval_env(
    run: RunDir,
    env_overrides: Mapping[str, Any] | None = None,
    game_length: int | None = None,
    force_eval_on_reserved: bool | None = None,
) -> EvalEnv:
    """Build the env `run` was trained on (with `env_overrides`), with episodes of `game_length` steps
    (default: the run's `general.game_length`)."""
    num_players = int(run.config['general']['num_players'])
    game_length = int(run.config['general']['game_length'] if game_length is None else game_length)
    env, env_mode, eval_on_reserved = make_env(
        env_config_from_mapping(env_section(run, env_overrides)), num_players, game_length, force_eval_on_reserved)
    return EvalEnv(env, env_mode, eval_on_reserved, num_players, game_length)


def eval_env_const(batched_env: Any, env_mode: EnvMode, eval_on_reserved: bool) -> Any:
    """The batched env's default const, switched to the held-out goal positions / layouts if requested."""
    flags = jnp.full((batched_env.num_envs,), eval_on_reserved, dtype=jnp.bool_)
    if env_mode == 'coop_foraging':
        return batched_env.default_const.replace(use_reserved_goal_positions=flags)
    return batched_env.default_const.replace(use_reserved_layouts=flags)


# -- policies ------------------------------------------------------------------------------------

def checkpoint_name(run_name: str, agent_id: int, ckpt: Checkpoint) -> str:
    return f'{run_name}_agent-{agent_id}_' + ('best' if ckpt == 'best' else f'step-{ckpt}')


@dataclass
class LoadedPolicy:
    run: RunDir
    agent_id: int
    ckpt: Checkpoint
    model_fn: Any
    model_state: PyTreeNode

    @property
    def name(self) -> str:
        return checkpoint_name(self.run.name, self.agent_id, self.ckpt)


class PolicyLoader:
    """Loads checkpoints from run directories, rebuilding each run's model only once."""

    def __init__(self) -> None:
        self._models: dict[Path, tuple[Any, PyTreeNode]] = {}

    def load(self, run: RunDir, agent_id: int, ckpt: Checkpoint = 'best') -> LoadedPolicy:
        key = run.path.resolve()
        if key not in self._models:
            self._models[key] = build_model_fn(run)
        model_fn, template = self._models[key]
        return LoadedPolicy(run, agent_id, ckpt, model_fn, run.load_params(agent_id, ckpt, template))

    def load_entry(self, entry: PartnerEntry, relative_to: RunDir | None = None) -> LoadedPolicy:
        return self.load(RunDir.open(resolve_run_path(entry.run, relative_to)), entry.agent_id, entry.ckpt)

    def load_population(self, run: RunDir, ckpt: Checkpoint = 'best',
                        agent_ids: Sequence[int] | None = None, max_agents: int | None = None) -> list[LoadedPolicy]:
        """All agents of a run (or `agent_ids`, or the first `max_agents`) at checkpoint `ckpt`."""
        ids = list(run.agent_ids()) if agent_ids is None else [int(i) for i in agent_ids]
        if max_agents is not None:
            ids = ids[:max_agents]
        if not ids:
            raise ValueError(f'No agents to load from {run.path}')
        policies = [self.load(run, i, ckpt) for i in ids]
        check_same_structure([p.model_state for p in policies], context=str(run.path))
        return policies


def check_same_structure(model_states: Sequence[PyTreeNode], context: str) -> None:
    """Raise unless all model states share one parameter structure (needed to reuse one compiled fn)."""
    ref = jax.tree_util.tree_structure(model_states[0])
    for i, state in enumerate(model_states[1:], start=1):
        if jax.tree_util.tree_structure(state) != ref:
            raise ValueError(f'Model state structure mismatch in {context} at index {i}.')


def model_signature(model_fn: Any, model_state: PyTreeNode) -> tuple:
    """Hashable key for caching compiled functions: model type, param tree structure and leaf shapes."""
    leaves = tuple((tuple(getattr(x, 'shape', ())), str(getattr(x, 'dtype', type(x).__name__)))
                   for x in jax.tree_util.tree_leaves(model_state))
    return type(model_fn).__name__, str(jax.tree_util.tree_structure(model_state)), leaves


# -- partner pools -------------------------------------------------------------------------------

STAGE_NAMES_3 = ('early', 'mid', 'late')


def stage_names(num_stages: int) -> tuple[str, ...]:
    return STAGE_NAMES_3 if num_stages == 3 else tuple(f'stage_{i}' for i in range(num_stages))


@dataclass
class PoolPartner:
    entry: PartnerEntry
    stage: int | None           # index of the checkpoint within its agent's selection; None for 'best'

    @property
    def name(self) -> str:
        return checkpoint_name(Path(self.entry.run).name, self.entry.agent_id, self.entry.ckpt)


@dataclass
class PartnerPoolConfig:
    runs: list[str] = field(default_factory=list)       # CEC run dirs; empty -> the ego run's partners.yaml
    num_checkpoints_per_agent: int | None = None        # None -> the ego run's partners.yaml
    include_best: bool = True                           # also include each partner agent's 'best' checkpoint


def label_stages(entries: Sequence[PartnerEntry]) -> list[PoolPartner]:
    """Label each step checkpoint with its index among its agent's selected steps (early ... late).

    Every partner agent must contribute the same number of step checkpoints, so a stage means the same
    point in training for every agent."""
    steps: dict[tuple[str, int], list[int]] = {}
    for e in entries:
        if e.ckpt != 'best':
            steps.setdefault((e.run, e.agent_id), []).append(int(e.ckpt))
    counts = {len(v) for v in steps.values()}
    if len(counts) > 1:
        raise ValueError(f'Partner agents have different numbers of step checkpoints {sorted(counts)}; '
                         'stages would not be comparable across agents.')
    return [PoolPartner(e, None if e.ckpt == 'best' else sorted(steps[(e.run, e.agent_id)]).index(int(e.ckpt)))
            for e in entries]


def partner_pool(partner_runs: Sequence[RunDir], num_checkpoints_per_agent: int, include_best: bool,
                 target_steps: Sequence[int] = ()) -> list[PoolPartner]:
    """Select partner checkpoints the same way train_cecp.py does (without a held-out split).

    Order: runs as given, agents ascending, 'best' first, then steps ascending. Evals that split one
    RNG key per partner depend on this order."""
    manifest = build_manifest(partner_runs, num_checkpoints_per_agent, include_best, target_steps, heldout_ratio=0.0)
    return label_stages(manifest.train)


def resolve_partner_pool(cfg: PartnerPoolConfig, ego_run: RunDir | None = None) -> tuple[list[PoolPartner], dict[str, Any]]:
    """Build the partner pool from `cfg`, filling unset fields from the ego run.

    Unset `runs`: a CECP ego's partners.yaml runs, or for a CEC population (no partners.yaml) the ego
    run itself. Unset `num_checkpoints_per_agent`: from partners.yaml. With a CECP ego and no explicit
    settings, the pool's step checkpoints (and so its stages) are exactly the ones its partner pool was
    built from. Returns (pool, the selection actually used)."""
    manifest = ego_run.read_partners() if ego_run is not None and (ego_run.path / PARTNERS_FILE).exists() else None
    if cfg.runs:
        run_paths = list(cfg.runs)
    elif manifest is not None:
        run_paths = [str(resolve_run_path(p, ego_run)) for p in manifest.partner_runs]
    elif ego_run is not None:
        run_paths = [str(ego_run.path)]   # a CEC population plays against its own checkpoints
    else:
        raise ValueError('Set partners.runs to the CEC run(s) to draw partners from.')
    if cfg.num_checkpoints_per_agent is not None:
        num_ckpts, target_steps = int(cfg.num_checkpoints_per_agent), []
    elif manifest is not None:
        num_ckpts, target_steps = manifest.num_checkpoints_per_agent, list(manifest.target_steps)
    else:
        raise ValueError(f'Set partners.num_checkpoints_per_agent ({ego_run.path if ego_run else "no ego run"} '
                         'has no partners.yaml to take it from).')
    pool = partner_pool([RunDir.open(p) for p in run_paths], num_ckpts, cfg.include_best, target_steps)
    selection = {'runs': run_paths, 'num_checkpoints_per_agent': num_ckpts, 'target_steps': target_steps,
                 'include_best': bool(cfg.include_best)}
    return pool, selection


# -- outputs -------------------------------------------------------------------------------------

def compute_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    stats = {'mean': float(np.mean(values)), 'std': float(np.std(values)),
             'min': float(np.min(values)), 'max': float(np.max(values))}
    for q in (5, 25, 50, 75, 95):
        stats[f'p{q:02d}'] = float(np.percentile(values, q))
    return stats


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dict rows; the header comes from the first row."""
    if not rows:
        raise ValueError(f'No rows to write to {path}')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: str | Path, matrix: np.ndarray, row_ids: Sequence[Any],
                     col_ids: Sequence[Any] | None = None, corner: str = 'agent_id') -> None:
    col_ids = row_ids if col_ids is None else col_ids
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([corner, *col_ids])
        for row_id, row in zip(row_ids, np.asarray(matrix)):
            writer.writerow([row_id, *row.tolist()])


def flatten_obs(obs: Any) -> jax.Array:
    """Concatenate all observation keys into one [batch, D] array."""
    parts = [obs[key].reshape(obs[key].shape[0], -1) for key in obs.keys()]
    return jnp.concatenate(parts, axis=-1)
