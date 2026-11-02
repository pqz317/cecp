"""Shared PPO training machinery for CEC (self-play population) and CECP (cooperator vs. frozen partners).

Both scripts train `general.num_agents` *learners* with PPO in a batched environment. CEC learners play
with themselves; CECP learners play with a pool of frozen partner policies. `TrainingController`
implements everything they share -- model/trainer construction, optional multi-GPU role sharding,
the compiled rollout + PPO update cycle, periodic evaluation, logging and checkpointing -- and a
subclass supplies the role layout, the frozen agents and the evaluation.

Roles in the training env are ordered `[learner_0, ..., learner_{L-1}, frozen_0, ..., frozen_{F-1}]`.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias
from warnings import warn

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .config import (
    ArchitectureConfig,
    CoopForagingEnvConfig,
    EnvConfig,
    OvercookedEnvConfig,
    PredictiveConfig,
    env_config_from_mapping,
    from_mapping,
)
from .controller.ippo import IPPOController
from .envs import Observation
from .envs.batched import FrozenAssignmentEnv
from .envs.make import make_env
from .policy.actor_critic import ActorCriticPPOAgent, ActorCriticPPOTrainer
from .policy.agent_wrapper import RecurrentAgent
from .policy.build import build_actor_critic
from .runs import RunDir
from .trainers.ppo import PPOTransition
from .utils import (
    build_role_to_shard_local,
    partition_roles_equal,
    pytree_repeat_stack,
    select_training_devices,
    split_rng_to_list,
)


# -- configs -------------------------------------------------------------------------------------

@dataclass
class GeneralConfig:
    seed: int = 0
    run_name: str = 'default'
    per_agent_env_steps: int = int(1e9)
    # CEC: size of the self-play population. CECP: number of independently trained cooperators.
    num_agents: int = 1
    num_players: int = 2
    game_length: int = 60
    eval_num_episodes: int = 32


@dataclass
class PPOConfig:
    rollout_length: int = 64
    per_agent_num_envs: int = 512
    lr: float = 1e-3
    lr_schedule: str = 'linear'
    lr_warmup_ratio: float = 0.0
    grad_clip_norm: float = 0.5
    ppo_epochs: int = 1
    # Per learner; minibatches per epoch = rollout_length * (learner batch) / batch_size.
    batch_size: int = 512
    minibatch_seq_len: int = 16
    gamma: float = 0.99
    gae_lam: float = 0.95
    pi_loss_coef: float = 1.0
    val_loss_coef: float = 0.3
    entropy_coef: float = 0.01
    ratio_clip: float = 0.2
    value_clip: float = 10.0


@dataclass
class LoggingConfig:
    runs_dir: str = 'runs'
    wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None

    def wandb_cfg(self) -> dict[str, Any] | None:
        if not self.wandb:
            return None
        return {'entity': self.wandb_entity, 'project': self.wandb_project, 'group': self.wandb_group, 'job_type': self.wandb_job_type}


@dataclass
class ShardingConfig:
    # Split the learners across all visible GPUs (one jax.pmap shard per device).
    enable_multi_gpu_training: bool = False


@dataclass
class TrainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    env: EnvConfig = field(default_factory=CoopForagingEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)

    def __post_init__(self) -> None:
        if self.ppo.lr_schedule not in ('constant', 'linear', 'warmup_cosine'):
            raise ValueError(f"ppo.lr_schedule must be 'constant', 'linear', or 'warmup_cosine', got '{self.ppo.lr_schedule}'")
        if self.general.num_agents < 1:
            raise ValueError(f'general.num_agents must be >= 1, got {self.general.num_agents}')


TRAIN_SECTIONS = ('general', 'env', 'ppo', 'architecture', 'predictive', 'logging', 'sharding')


def train_config_from_mapping(cfg: Mapping[str, Any]) -> TrainConfig:
    """Build a TrainConfig from the training sections of a (resolved) config mapping."""
    return TrainConfig(
        general=from_mapping(GeneralConfig, cfg['general']),
        env=env_config_from_mapping(cfg['env']),
        ppo=from_mapping(PPOConfig, cfg['ppo']),
        architecture=from_mapping(ArchitectureConfig, cfg['architecture']),
        predictive=from_mapping(PredictiveConfig, cfg['predictive']),
        logging=from_mapping(LoggingConfig, cfg.get('logging', {})),
        sharding=from_mapping(ShardingConfig, cfg.get('sharding', {})),
    )


def resolve_hydra_config(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


# -- helpers -------------------------------------------------------------------------------------

_REDUCTIONS = {'mean': jnp.mean, 'std': jnp.std, 'min': jnp.min, 'max': jnp.max}


def aggregate_stats(info_lst: Sequence[Mapping[str, float]], reductions: Sequence[str] = ('mean', 'std', 'min', 'max')) -> dict[str, float]:
    """Reduce each scalar key across learners (default: mean/std/min/max)."""
    stats: dict[str, float] = {}
    for reduce_name in reductions:
        reduce_fn = _REDUCTIONS[reduce_name]
        for k in info_lst[0].keys():
            stats[f'{k}_{reduce_name}'] = reduce_fn(jnp.array([info[k] for info in info_lst])).item()
    return stats


RunState: TypeAlias = tuple[jax.Array, list[PyTreeNode], list[PyTreeNode], PyTreeNode, list[Observation]]


class TrainingController(IPPOController):
    """Trains `general.num_agents` learners; see the module docstring. Subclasses implement:

      - `train_env_agent_to_role(num_learners)`: role id per (env, seat) for a training env holding
        `num_learners` learners (local ids 0..num_learners-1, frozen roles after them), with
        `per_agent_num_envs` envs per learner.
      - `init_eval()` and `evaluate(learner_id, model_state) -> (log_info, score)`; `best` checkpoints
        track the highest `score`.

    and may set `frozen_agent_lst` / `frozen_model_state_lst` (before calling `setup`), and override
    `learner_log_name(i)`, `train_log_prefix`, `eval_log_prefix`, `eval_aggregate_reductions` and
    `extra_log_info`.
    """
    frozen_agent_lst: list[RecurrentAgent] = []
    frozen_model_state_lst: list[PyTreeNode] = []
    train_log_prefix = 'train'
    eval_log_prefix = 'eval'
    eval_aggregate_reductions: tuple[str, ...] = ('mean', 'std', 'min', 'max')

    def learner_log_name(self, learner_id: int) -> str:
        return f'agent#{learner_id}'

    @property
    def extra_log_info(self) -> dict[str, float]:
        return {}

    def setup(self, config: TrainConfig, rundir: RunDir) -> None:
        """Build env, models, evaluation and the compiled training cycle, then attach to `rundir`."""
        general, ppo_cfg = config.general, config.ppo
        self.config = config
        self.num_learners: int = general.num_agents
        self.num_players: int = general.num_players
        self.game_length: int = general.game_length
        self.per_agent_env_steps: int = general.per_agent_env_steps

        self.env_fn, self.env_mode, self.eval_on_reserved_envs = make_env(config.env, general.num_players, general.game_length)
        if isinstance(config.env, OvercookedEnvConfig):
            self.reward_shaping_steps = config.env.reward_shaping_steps
            self.train_shaped_reward_factor = config.env.train_shaped_reward_factor
        else:
            self.reward_shaping_steps, self.train_shaped_reward_factor = 0, 0.0

        self.use_predictive_objective: bool = config.predictive.use_predictive_objective
        self.obs_space = self.env_fn.get_observation_space()[0]
        self.action_space = self.env_fn.get_action_space()[0]
        self.model_fn_lst = [
            build_actor_critic(config.architecture, config.predictive, self.obs_space, self.action_space,
                               output_nn_depth=5 if i == 0 else -1)  # print the network summary once
            for i in range(self.num_learners)
        ]

        self.rng_train = jax.random.key(general.seed + int(1e8))
        self.rng_eval = jax.random.key(general.seed + int(2e8))
        self.eval_num_episodes: int = general.eval_num_episodes
        self.eval_step_interval_per_agent: int = max(1, general.per_agent_env_steps // 50)  # evaluate 50 times over training
        self.init_eval()

        self.train_rollout_length: int = ppo_cfg.rollout_length
        self.train_per_agent_num_envs: int = ppo_cfg.per_agent_num_envs
        self.train_num_envs: int = self.train_per_agent_num_envs * self.num_learners
        self._setup_role_shards(config.sharding.enable_multi_gpu_training)

        role_batch_sizes = self.get_train_batched_env(tuple(range(self.num_learners))).get_agent_batch_size()[:self.num_learners]
        role_batch_size = role_batch_sizes[0]
        if any(b != role_batch_size for b in role_batch_sizes):
            warn(f'Learner batch sizes are not uniform ({role_batch_sizes}); using the first for the LR schedule.')
        minibatch_num_chunks = ppo_cfg.batch_size // ppo_cfg.minibatch_seq_len
        if ppo_cfg.batch_size % ppo_cfg.minibatch_seq_len != 0:
            warn(f'Batch size {ppo_cfg.batch_size} is not a multiple of chunk length {ppo_cfg.minibatch_seq_len}. '
                 f'It will be adjusted to {minibatch_num_chunks * ppo_cfg.minibatch_seq_len} to fit evenly.')
        if ppo_cfg.rollout_length % ppo_cfg.minibatch_seq_len != 0:
            warn(f'Training rollout length {ppo_cfg.rollout_length} is not a multiple of chunk length {ppo_cfg.minibatch_seq_len}. '
                 f'It will be truncated to {ppo_cfg.rollout_length // ppo_cfg.minibatch_seq_len * ppo_cfg.minibatch_seq_len} for model updates.')
        # Total optimizer steps, for the LR schedule.
        lr_schedule_steps = self.per_agent_env_steps * ppo_cfg.ppo_epochs * role_batch_size // (self.train_per_agent_num_envs * ppo_cfg.batch_size) + 1
        self.ppo_trainer_lst = [
            ActorCriticPPOTrainer(
                actor_critic_model=model_fn,
                lr=ppo_cfg.lr,
                lr_schedule_steps=lr_schedule_steps,
                lr_schedule=ppo_cfg.lr_schedule,
                lr_warmup_ratio=ppo_cfg.lr_warmup_ratio,
                grad_clip_norm=ppo_cfg.grad_clip_norm,
                pred_loss_coef=config.predictive.pred_loss_coef if self.use_predictive_objective else 0.0,
                pi_loss_coef=ppo_cfg.pi_loss_coef,
                val_loss_coef=ppo_cfg.val_loss_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                gamma=ppo_cfg.gamma,
                gae_lam=ppo_cfg.gae_lam,
                ppo_epochs=ppo_cfg.ppo_epochs,
                ratio_clip=ppo_cfg.ratio_clip,
                value_clip=ppo_cfg.value_clip,
                minibatch_seq_len=ppo_cfg.minibatch_seq_len,
                minibatch_num_chunks=minibatch_num_chunks,
            )
            for model_fn in self.model_fn_lst
        ]
        self.init_train()
        self.init_run(rundir, config.logging.wandb_cfg())

    # -- evaluation helpers ----------------------------------------------------------------------

    def compile_eval_rollout(self, eval_env_fn, eval_agent_lst: Sequence[RecurrentAgent], model_state_lst_example: Sequence[PyTreeNode]):
        """AOT-compile `(rng, model_state_lst) -> reward_seq_lst` for one batch of evaluation episodes
        (reserved goals/layouts if `eval_on_reserved_envs`)."""
        if self.env_mode == 'coop_foraging':
            env_const = eval_env_fn.default_const.replace(
                use_reserved_goal_positions=jnp.full((eval_env_fn.num_envs,), self.eval_on_reserved_envs, dtype=jnp.bool_))
        else:
            env_const = eval_env_fn.default_const.replace(
                use_reserved_layouts=jnp.full((eval_env_fn.num_envs,), self.eval_on_reserved_envs, dtype=jnp.bool_))

        def eval_episode(rng: jax.Array, model_state_lst: Sequence[PyTreeNode]) -> list[jax.Array]:
            def step(carry, _):
                rng, agent_state_lst, env_state, obs_lst = carry
                rng, next_agent_state_lst, next_env_state, next_obs_lst, _, reward_lst, _, _ = self.rollout_step(
                    env_fn=eval_env_fn, agent_lst=eval_agent_lst, rng=rng, model_state_lst=model_state_lst,
                    agent_state_lst=agent_state_lst, env_const=env_const, env_state=env_state, obs_lst=obs_lst,
                )
                return (rng, next_agent_state_lst, next_env_state, next_obs_lst), reward_lst

            rng, rng_init_agent_lst = split_rng_to_list(rng, len(eval_agent_lst))
            agent_state_lst = [
                agent_fn.init_agent_state(rng_init_agent, batch_size)
                for rng_init_agent, agent_fn, batch_size in zip(rng_init_agent_lst, eval_agent_lst, eval_env_fn.get_agent_batch_size())
            ]
            rng, rng_reset = jax.random.split(rng)
            env_state, obs_lst = eval_env_fn.reset(rng_reset, env_const)
            _, reward_seq_lst = jax.lax.scan(step, (rng, agent_state_lst, env_state, obs_lst), None, length=self.game_length)
            return reward_seq_lst

        return jax.jit(eval_episode).lower(jax.random.key(0), list(model_state_lst_example)).compile()

    # -- training env ------------------------------------------------------------------------------

    def get_train_batched_env(self, learner_ids: tuple[int, ...]) -> FrozenAssignmentEnv:
        """Training env for one shard of learners (all learners when not sharding)."""
        if not hasattr(self, '_train_env_cache'):
            self._train_env_cache: dict[tuple[int, ...], FrozenAssignmentEnv] = {}
        if learner_ids not in self._train_env_cache:
            env_agent_to_role = self.train_env_agent_to_role(len(learner_ids))
            self._train_env_cache[learner_ids] = FrozenAssignmentEnv(self.env_fn, len(env_agent_to_role), env_agent_to_role=env_agent_to_role)
        return self._train_env_cache[learner_ids]

    def get_train_env_const_step(self, per_agent_env_step: int, num_envs: int) -> PyTreeNode:
        """Training env constants: never the reserved goals/layouts; Overcooked's shaped reward is
        annealed linearly to 0 over `env.reward_shaping_steps` per-agent steps."""
        if self.env_mode == 'coop_foraging':
            env_const = self.env_fn.default_const.replace(use_reserved_goal_positions=False)
        else:
            shaping_factor = max(0.0, 1.0 - per_agent_env_step / self.reward_shaping_steps) if self.reward_shaping_steps > 0 else 0.0
            env_const = self.env_fn.default_const.replace(
                use_reserved_layouts=False,
                shaped_reward_factor=self.train_shaped_reward_factor * shaping_factor,
            )
        return pytree_repeat_stack(env_const, (num_envs,))

    def get_train_env_const_seq(self, per_agent_env_step: int, num_envs: int) -> PyTreeNode:
        return pytree_repeat_stack(self.get_train_env_const_step(per_agent_env_step, num_envs), (self.train_rollout_length,))

    # -- sharding --------------------------------------------------------------------------------

    def _setup_role_shards(self, enable_multi_gpu: bool) -> None:
        self.enable_multi_gpu_training = enable_multi_gpu
        self.training_devices = select_training_devices('gpu')
        if enable_multi_gpu and len(self.training_devices) < 2:
            raise ValueError(f'sharding.enable_multi_gpu_training=True requires at least 2 devices, got {len(self.training_devices)}')
        if enable_multi_gpu:
            self.role_shards = partition_roles_equal(self.num_learners, len(self.training_devices))
            self.roles_per_shard = len(self.role_shards[0])
            if any(len(role_ids) != self.roles_per_shard for role_ids in self.role_shards):
                raise AssertionError(f'all shards must hold the same number of learners, got {[len(r) for r in self.role_shards]}')
            self.role_to_shard_local = build_role_to_shard_local(self.role_shards)
        else:
            self.role_shards = [list(range(self.num_learners))]
            self.roles_per_shard = self.num_learners
            self.role_to_shard_local = {role_id: (0, role_id) for role_id in range(self.num_learners)}
        self.num_training_shards = len(self.role_shards)
        self.shard_num_envs = self.train_per_agent_num_envs * self.roles_per_shard

    # -- compiled training cycle -----------------------------------------------------------------

    def init_train(self) -> None:
        learner_ids = tuple(self.role_shards[0])  # every shard has the same layout, so shard 0 is the template
        train_env_fn = self.get_train_batched_env(learner_ids)
        learner_agent_lst = [ActorCriticPPOAgent(self.model_fn_lst[i]) for i in learner_ids]
        agent_lst = learner_agent_lst + list(self.frozen_agent_lst)
        trainer_lst = [self.ppo_trainer_lst[i] for i in learner_ids]
        num_local = len(learner_ids)
        # RNG keys for trainer init: one per agent on a single device (frozen agents' keys unused),
        # one per learner per shard when sharded -- as in the paper runs.
        num_trainer_rng = len(agent_lst) if not self.enable_multi_gpu_training else num_local

        def init_run_state(rng: jax.Array) -> RunState:
            rng, rng_trainer_lst = split_rng_to_list(rng, num_trainer_rng)
            trainer_state_lst = [trainer_fn.init_trainer_state(r) for r, trainer_fn in zip(rng_trainer_lst, trainer_lst)]
            rng, rng_init_agent_lst = split_rng_to_list(rng, len(agent_lst))
            agent_state_lst = [
                agent_fn.init_agent_state(r, batch_size)
                for r, agent_fn, batch_size in zip(rng_init_agent_lst, agent_lst, train_env_fn.get_agent_batch_size())
            ]
            rng, rng_reset = jax.random.split(rng)
            env_state, obs_lst = train_env_fn.reset(rng_reset, self.get_train_env_const_step(0, self.shard_num_envs))
            return rng, trainer_state_lst, agent_state_lst, env_state, obs_lst

        def ppo_training_cycle(run_state: RunState, frozen_model_state_lst: Sequence[PyTreeNode], env_const_seq: PyTreeNode):
            """One rollout of `rollout_length` steps in every env, then a PPO update of every learner."""
            rng, trainer_state_lst, agent_state_lst, env_state, last_obs_lst = run_state
            learner_model_state_lst = [t.model_state_from_trainer_state(s) for t, s in zip(trainer_lst, trainer_state_lst)]
            model_state_lst = learner_model_state_lst + list(frozen_model_state_lst)

            def step(carry, env_const):
                rng, agent_state_lst, env_state, obs_lst = carry
                rng, next_agent_state_lst, next_env_state, next_obs_lst, data_lst, info = self.ppo_rollout_step(
                    train_env_fn, agent_lst, rng, model_state_lst, agent_state_lst, env_const, env_state, obs_lst,
                )
                return (rng, next_agent_state_lst, next_env_state, next_obs_lst), (data_lst, info)

            (rng, agent_state_lst, env_state, last_obs_lst), (data_lst, _) = jax.lax.scan(
                step, (rng, agent_state_lst, env_state, last_obs_lst), env_const_seq, length=self.train_rollout_length,
            )
            learner_data_lst = data_lst[:num_local]
            env_info_lst = [{'avg-return': data['reward'].mean() * self.train_rollout_length} for data in learner_data_lst]
            buffer_lst = [
                PPOTransition(data['obs'], data['action'], data['reward'], data['done'], data['log_p'], data['val'])
                for data in learner_data_lst
            ]
            last_val_lst = []
            rng, rng_val_lst = split_rng_to_list(rng, num_local)
            for rng_val, agent_fn, model_state, agent_state, last_obs in zip(
                rng_val_lst, learner_agent_lst, learner_model_state_lst, agent_state_lst[:num_local], last_obs_lst[:num_local],
            ):
                _, _, extra_data = agent_fn.step(rng_val, model_state, agent_state, last_obs)
                last_val_lst.append(extra_data['val'])
            aux_data_lst = []
            for data in learner_data_lst:
                aux_data = {'agent_state': data['agent_state']}
                if self.use_predictive_objective:
                    aux_data['next_obs'] = data['next_obs']  # target for the next-state prediction loss
                aux_data_lst.append(aux_data)
            rng, rng_trainer = jax.random.split(rng)
            new_trainer_state_lst, optim_info_lst = self.ppo_update(trainer_lst, rng_trainer, trainer_state_lst, buffer_lst, last_val_lst, aux_data_lst)
            return (rng, new_trainer_state_lst, agent_state_lst, env_state, last_obs_lst), env_info_lst, optim_info_lst

        start_time = time.perf_counter()
        env_const_seq_example = self.get_train_env_const_seq(0, self.shard_num_envs)
        if not self.enable_multi_gpu_training:
            print('jax compilation of the [ppo_training_cycle] function...')
            self.init_run_state = init_run_state
            self.training_cycle = jax.jit(ppo_training_cycle).lower(
                init_run_state(jax.random.key(0)), list(self.frozen_model_state_lst), env_const_seq_example,
            ).compile()
        else:
            print(f'jax compilation of the [pmapped_ppo_training_cycle] function: {self.num_training_shards} shards over '
                  f'{len(self.training_devices)} device(s), learner shards={self.role_shards}...')

            def init_run_state_pmapped(rng: jax.Array) -> RunState:
                rng, rng_shard_lst = split_rng_to_list(rng, self.num_training_shards)
                run_state_lst = [init_run_state(r) for r in rng_shard_lst]
                return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *run_state_lst)

            self.init_run_state = init_run_state_pmapped
            self.training_cycle = jax.pmap(
                ppo_training_cycle, in_axes=(0, None, 0), out_axes=(0, 0, 0), devices=self.training_devices,
            ).lower(
                init_run_state_pmapped(jax.random.key(0)), list(self.frozen_model_state_lst), self._broadcast_shards(env_const_seq_example),
            ).compile()
        print(f'Compilation completed in {time.perf_counter() - start_time:.2f} seconds.')

    def _broadcast_shards(self, tree: PyTreeNode) -> PyTreeNode:
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (self.num_training_shards,) + x.shape), tree)

    def _unshard(self, per_local_role: Sequence[PyTreeNode]) -> list[PyTreeNode]:
        """Per-learner values from pmapped per-local-role outputs (leading axis = shard)."""
        out = []
        for learner_id in range(self.num_learners):
            shard_idx, local_idx = self.role_to_shard_local[learner_id]
            out.append(jax.tree_util.tree_map(lambda x: x[shard_idx], per_local_role[local_idx]))
        return out

    # -- main loop -------------------------------------------------------------------------------

    def train(self) -> None:
        run_state = self.init_run_state(self.rng_train)
        per_agent_env_step = 0
        steps_since_last_write = 0
        steps_since_last_eval = self.eval_step_interval_per_agent  # evaluate after the first cycle
        best_score = [-1e9] * self.num_learners
        step_increase = self.train_rollout_length * self.train_per_agent_num_envs

        with tqdm(total=self.per_agent_env_steps, desc='Train', unit='frames', position=2, ncols=80) as pbar:
            while per_agent_env_step < self.per_agent_env_steps:
                if not self.enable_multi_gpu_training:
                    env_const_seq = self.get_train_env_const_seq(per_agent_env_step, self.shard_num_envs)
                    run_state, train_info_lst, optim_info_lst = self.training_cycle(run_state, list(self.frozen_model_state_lst), env_const_seq)
                else:
                    env_const_seq = self._broadcast_shards(self.get_train_env_const_seq(per_agent_env_step, self.shard_num_envs))
                    run_state, train_info_lst, optim_info_lst = self.training_cycle(run_state, list(self.frozen_model_state_lst), env_const_seq)
                    train_info_lst, optim_info_lst = self._unshard(train_info_lst), self._unshard(optim_info_lst)

                per_agent_env_step += step_increase
                steps_since_last_write += step_increase
                steps_since_last_eval += step_increase
                pbar.update(step_increase)

                info_lst = []
                for i in range(self.num_learners):
                    name = self.learner_log_name(i)
                    info_lst.append(self.itemize_dict(train_info_lst[i]))
                    self.log(train_info_lst[i] | self.extra_log_info, per_agent_env_step, log_prefix=f'{name}_train')
                    self.log(optim_info_lst[i] | self.extra_log_info, per_agent_env_step, log_prefix=f'{name}_optimization')
                train_info_aggr = aggregate_stats(info_lst)
                self.log(train_info_aggr | self.extra_log_info, per_agent_env_step, log_prefix=self.train_log_prefix)
                if steps_since_last_write > 10_000:
                    steps_since_last_write -= 10_000
                    tqdm.write(f"rollout-chunk return at step {per_agent_env_step}: {train_info_aggr['avg-return_mean']:.3f} "
                               f"(std {train_info_aggr['avg-return_std']:.3f})")

                if steps_since_last_eval > self.eval_step_interval_per_agent:
                    steps_since_last_eval -= self.eval_step_interval_per_agent
                    tqdm.write('\n' + '=' * 20 + f' per_agent_env_step: {per_agent_env_step} / {self.per_agent_env_steps} ' + '=' * 20)
                    trainer_state_lst = run_state[1] if not self.enable_multi_gpu_training else self._unshard(run_state[1])
                    eval_info_lst = []
                    for i in range(self.num_learners):
                        model_state = self.ppo_trainer_lst[i].model_state_from_trainer_state(trainer_state_lst[i])
                        eval_info, score = self.evaluate(i, model_state)
                        eval_info_lst.append(eval_info)
                        self.log(eval_info | self.extra_log_info, per_agent_env_step, log_prefix=f'{self.learner_log_name(i)}_eval')
                        self.rundir.save_params(i, per_agent_env_step, model_state)
                        if score > best_score[i]:
                            best_score[i] = score
                            self.rundir.save_params(i, 'best', model_state)
                    eval_info_aggr = aggregate_stats(eval_info_lst, self.eval_aggregate_reductions)
                    tqdm.write(f'eval: {eval_info_aggr}')
                    self.log(eval_info_aggr | self.extra_log_info, per_agent_env_step, log_prefix=self.eval_log_prefix)
        self.metrics.close()

    # -- subclass hooks --------------------------------------------------------------------------

    def train_env_agent_to_role(self, num_learners: int) -> list[list[int]]:
        raise NotImplementedError

    def init_eval(self) -> None:
        raise NotImplementedError

    def evaluate(self, learner_id: int, model_state: PyTreeNode) -> tuple[dict[str, float], float]:
        raise NotImplementedError


def run_config_dict(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The resolved config to store as the run's config.yaml (Hydra's own section dropped)."""
    return {k: v for k, v in cfg.items() if k != 'hydra'}


def create_rundir(train_cfg: TrainConfig, resolved_cfg: Mapping[str, Any]) -> RunDir:
    return RunDir.create(Path(train_cfg.logging.runs_dir) / train_cfg.general.run_name, run_config_dict(resolved_cfg))
