"""Local run directories: resolved config, checkpoints, partner manifest and metrics.

Layout of a run directory::

    runs/<name>/
      config.yaml                 # full resolved config of the run
      metrics.csv                 # long format: step,key,value
      partners.yaml               # CECP runs only: which partner checkpoints were trained against / held out
      agents/agent_<i>/best/params.msgpack
      agents/agent_<i>/step_<n>/params.msgpack

Checkpoints hold parameters only. Models are rebuilt from config.yaml (see `load_policy`), so a run
directory never depends on Python module paths the way pickled models do.
"""
import csv
import dataclasses
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization
from flax.struct import PyTreeNode

Checkpoint = Literal['best'] | int

CONFIG_FILE = 'config.yaml'
METRICS_FILE = 'metrics.csv'
PARTNERS_FILE = 'partners.yaml'
PARAMS_FILE = 'params.msgpack'

_AGENT_DIR_RE = re.compile(r'^agent_(\d+)$')
_STEP_DIR_RE = re.compile(r'^step_(\d+)$')


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _ckpt_dir_name(ckpt: Checkpoint) -> str:
    if ckpt == 'best':
        return 'best'
    if isinstance(ckpt, (int, np.integer)) and not isinstance(ckpt, bool) and ckpt >= 0:
        return f'step_{int(ckpt)}'
    raise ValueError(f"checkpoint must be 'best' or a non-negative int step, got {ckpt!r}")


class RunDir:
    def __init__(self, path: str | os.PathLike, config: dict[str, Any]) -> None:
        self.path = Path(path)
        self.config = config

    @classmethod
    def create(cls, path: str | os.PathLike, config: Mapping[str, Any], exist_ok: bool = False) -> 'RunDir':
        """Create a run directory and write its resolved config. `config` is a plain mapping
        (e.g. `OmegaConf.to_container(cfg, resolve=True)` or `dataclasses.asdict(...)`)."""
        path = Path(path)
        if (path / CONFIG_FILE).exists() and not exist_ok:
            raise FileExistsError(f'Run directory {path} already exists; pick a new run name or pass exist_ok=True.')
        config = _to_plain(config)
        _atomic_write_bytes(path / CONFIG_FILE, yaml.safe_dump(config, sort_keys=False).encode())
        return cls(path, config)

    @classmethod
    def open(cls, path: str | os.PathLike) -> 'RunDir':
        path = Path(path)
        config_path = path / CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(f'{config_path} not found; is {path} a run directory?')
        with open(config_path) as f:
            return cls(path, yaml.safe_load(f))

    @property
    def name(self) -> str:
        return self.path.name

    def __repr__(self) -> str:
        return f'RunDir({str(self.path)!r})'

    # -- checkpoints -----------------------------------------------------------------------------

    def ckpt_dir(self, agent_id: int, ckpt: Checkpoint) -> Path:
        return self.path / 'agents' / f'agent_{int(agent_id)}' / _ckpt_dir_name(ckpt)

    def save_params(self, agent_id: int, ckpt: Checkpoint, model_state: PyTreeNode) -> Path:
        """Save a model state (e.g. `{'actor_critic_params': params}`) as msgpack."""
        path = self.ckpt_dir(agent_id, ckpt) / PARAMS_FILE
        _atomic_write_bytes(path, serialization.to_bytes(jax.device_get(model_state)))
        return path

    def load_params(self, agent_id: int, ckpt: Checkpoint, template: PyTreeNode) -> PyTreeNode:
        """Restore a model state saved by `save_params` into the structure of `template`
        (typically `model_fn.init_model_state(jax.random.key(0))`)."""
        path = self.ckpt_dir(agent_id, ckpt) / PARAMS_FILE
        if not path.exists():
            raise FileNotFoundError(f'No checkpoint at {path}')
        return serialization.from_bytes(template, path.read_bytes())

    def agent_ids(self) -> list[int]:
        agents_dir = self.path / 'agents'
        if not agents_dir.is_dir():
            return []
        ids = [int(m.group(1)) for p in agents_dir.iterdir() if (m := _AGENT_DIR_RE.match(p.name))]
        return sorted(ids)

    def list_checkpoints(self, agent_id: int) -> dict[str, Any]:
        """Returns {'best': Path | None, 'steps': [(step, Path), ...] sorted by step}."""
        agent_dir = self.path / 'agents' / f'agent_{int(agent_id)}'
        best = agent_dir / 'best' / PARAMS_FILE
        steps: list[tuple[int, Path]] = []
        if agent_dir.is_dir():
            for p in agent_dir.iterdir():
                m = _STEP_DIR_RE.match(p.name)
                if m and (p / PARAMS_FILE).exists():
                    steps.append((int(m.group(1)), p / PARAMS_FILE))
        return {'best': best if best.exists() else None, 'steps': sorted(steps)}

    # -- partner manifest ------------------------------------------------------------------------

    def write_partners(self, manifest: 'PartnerManifest') -> None:
        _atomic_write_bytes(self.path / PARTNERS_FILE, yaml.safe_dump(manifest.to_dict(), sort_keys=False).encode())

    def read_partners(self) -> 'PartnerManifest':
        path = self.path / PARTNERS_FILE
        if not path.exists():
            raise FileNotFoundError(f'{path} not found; only CECP (cooperator) runs have a partner manifest.')
        with open(path) as f:
            return PartnerManifest.from_dict(yaml.safe_load(f))


def _to_plain(obj: Any) -> Any:
    """Convert dataclasses / tuples / numpy scalars into YAML-safe builtins."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_plain(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# -- partner manifest ----------------------------------------------------------------------------

@dataclass
class PartnerEntry:
    run: str            # path of the partner (CEC population) run directory
    agent_id: int
    ckpt: Checkpoint    # 'best' or a step

    def to_dict(self) -> dict[str, Any]:
        return {'run': self.run, 'agent_id': int(self.agent_id), 'ckpt': self.ckpt if self.ckpt == 'best' else int(self.ckpt)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'PartnerEntry':
        ckpt = d['ckpt']
        return cls(run=str(d['run']), agent_id=int(d['agent_id']), ckpt=ckpt if ckpt == 'best' else int(ckpt))


@dataclass
class PartnerManifest:
    """The partner pool a cooperator was trained with, and which partner agents were held out."""
    partner_runs: list[str]
    num_checkpoints_per_agent: int
    include_best: bool
    target_steps: list[int]         # if non-empty, the checkpoints nearest these steps were used
    heldout_ratio: float
    heldout_seed: int
    train: list[PartnerEntry] = field(default_factory=list)
    heldout: list[PartnerEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in ('train', 'heldout')}
        d['target_steps'] = [int(s) for s in self.target_steps]
        d['train'] = [e.to_dict() for e in self.train]
        d['heldout'] = [e.to_dict() for e in self.heldout]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'PartnerManifest':
        d = dict(d)
        d['train'] = [PartnerEntry.from_dict(e) for e in d.get('train', [])]
        d['heldout'] = [PartnerEntry.from_dict(e) for e in d.get('heldout', [])]
        return cls(**d)


# -- metrics -------------------------------------------------------------------------------------

def itemize_dict(data: dict[str, Any]) -> dict[str, int | float]:
    """Reduce logged values to Python scalars (arrays and lists are averaged)."""
    res = {}
    for k, v in data.items():
        if isinstance(v, int) or isinstance(v, float):
            pass
        elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = np.mean(v).item() if v.size > 1 else v.item()
        elif isinstance(v, jax.Array):
            v = jnp.mean(v).item() if v.size > 1 else v.item()
        elif isinstance(v, list):
            v = np.mean(v).item()
        else:
            raise TypeError(f"log type {type(v)} of {v} not supported")
        res[k] = v
    return res


class MetricsLogger:
    """Appends metrics to <run>/metrics.csv (columns step,key,value) and optionally to W&B."""

    def __init__(self, run: RunDir, wandb_cfg: Mapping[str, Any] | None = None) -> None:
        self.path = run.path / METRICS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'key', 'value'])
        self._wandb = None
        if wandb_cfg is not None:
            import wandb  # optional dependency
            wandb.init(**{'config': run.config, 'name': run.name, **dict(wandb_cfg)})
            self._wandb = wandb

    def log(self, info: dict[str, Any], step: int | None = None, log_prefix: str | None = None) -> None:
        info = itemize_dict(info)
        if log_prefix is not None:
            info = {f'{log_prefix}/{k}': v for k, v in info.items()}
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            for k, v in info.items():
                writer.writerow(['' if step is None else int(step), k, v])
        if self._wandb is not None:
            self._wandb.log(info, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None


# -- rebuilding a policy -------------------------------------------------------------------------

def build_model_fn(run: RunDir | str | os.PathLike) -> tuple[Any, PyTreeNode]:
    """Rebuild the run's model from its config.yaml, without loading a checkpoint.

    Uses the config sections `env`, `general.num_players`, `general.game_length`, `architecture` and
    `predictive`. Returns (model_fn, template): all agents of a run share the model_fn, and `template`
    is a freshly initialised model state to pass to `RunDir.load_params`.
    """
    from .config import ArchitectureConfig, PredictiveConfig, env_config_from_mapping, from_mapping
    from .envs.make import make_env
    from .policy.build import build_actor_critic

    if not isinstance(run, RunDir):
        run = RunDir.open(run)
    cfg = run.config
    env, _, _ = make_env(
        env_config_from_mapping(cfg['env']),
        num_players=int(cfg['general']['num_players']),
        game_length=int(cfg['general']['game_length']),
    )
    model_fn = build_actor_critic(
        from_mapping(ArchitectureConfig, cfg['architecture']),
        from_mapping(PredictiveConfig, cfg['predictive']),
        env.get_observation_space()[0],
        env.get_action_space()[0],
    )
    return model_fn, model_fn.init_model_state(jax.random.key(0))


def load_policy(run: RunDir | str | os.PathLike, agent_id: int, ckpt: Checkpoint = 'best') -> tuple[Any, PyTreeNode]:
    """Rebuild agent `agent_id`'s model from the run's config.yaml and load checkpoint `ckpt`.

    Returns (model_fn, model_state), as consumed by `ActorWrapperAgent` and friends. To load many
    checkpoints of one run, call `build_model_fn` once and then `RunDir.load_params` per checkpoint.
    """
    if not isinstance(run, RunDir):
        run = RunDir.open(run)
    model_fn, template = build_model_fn(run)
    return model_fn, run.load_params(agent_id, ckpt, template)
