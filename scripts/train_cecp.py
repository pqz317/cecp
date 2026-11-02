"""Train CECP cooperators against a frozen partner pool (paper Sec. 4, Stage 2).

The partner pool is built from one or more CEC population runs (train_cec.py): every partner agent
contributes its 'best' checkpoint and `partners.num_checkpoints_per_agent` training checkpoints
(early ... late), and a fraction `partners.heldout_ratio` of partner *agents* is held out for
evaluation. The selection is written to <run>/partners.yaml. With `predictive.use_predictive_objective`
the cooperator also learns to predict the next observation (paper Eq. 6-7).

    python scripts/train_cec.py experiment=cf_cec
    python scripts/train_cecp.py experiment=cf_cecp            # partners.runs=[runs/cf_cec]
"""
from dataclasses import dataclass, field

import hydra
import jax
from flax.struct import PyTreeNode
from omegaconf import DictConfig

from cecp.config import from_mapping
from cecp.envs.batched import FrozenAssignmentEnv, HomogeneousBatchedEnv
from cecp.partners import build_manifest
from cecp.policy.actor_critic import ActorWrapperAgent
from cecp.runs import PartnerEntry, RunDir, load_policy
from cecp.training import TrainConfig, TrainingController, create_rundir, resolve_hydra_config, train_config_from_mapping


@dataclass
class PartnersConfig:
    runs: list[str] = field(default_factory=list)   # CEC run directories to draw partners from
    num_checkpoints_per_agent: int = 3               # evenly spaced training checkpoints per partner agent
    include_best: bool = True                        # also include each partner agent's 'best' checkpoint
    target_steps: list[int] = field(default_factory=list)  # if set, use the checkpoints nearest these steps instead
    heldout_ratio: float = 0.2                       # fraction of partner agents held out for evaluation
    heldout_seed: int = 0


def _load_partner_agents(entries: list[PartnerEntry]) -> tuple[list[ActorWrapperAgent], list[PyTreeNode]]:
    agents, states = [], []
    for entry in entries:
        model_fn, model_state = load_policy(entry.run, entry.agent_id, entry.ckpt)
        agents.append(ActorWrapperAgent(model_fn))
        states.append(model_state)
    return agents, states


class CECPController(TrainingController):
    train_log_prefix = 'cooperator_train'
    eval_log_prefix = 'cooperator_eval'
    eval_aggregate_reductions = ('mean',)  # each cooperator's eval info already has std/min/max over episodes

    def __init__(self, config: TrainConfig, partners_cfg: PartnersConfig, rundir: RunDir) -> None:
        if not partners_cfg.runs:
            raise ValueError('partners.runs is empty: pass the CEC run directories to train against, e.g. partners.runs=[runs/cf_cec]')
        manifest = build_manifest(
            [RunDir.open(p) for p in partners_cfg.runs],
            num_checkpoints_per_agent=partners_cfg.num_checkpoints_per_agent,
            include_best=partners_cfg.include_best,
            target_steps=partners_cfg.target_steps,
            heldout_ratio=partners_cfg.heldout_ratio,
            heldout_seed=partners_cfg.heldout_seed,
        )
        rundir.write_partners(manifest)
        self.frozen_agent_lst, self.frozen_model_state_lst = _load_partner_agents(manifest.train)
        # Evaluate (and select 'best') against the held-out partners when there are any, as in the paper runs;
        # these are the same partners later used for the novel-partner evaluation.
        eval_entries = manifest.heldout if manifest.heldout else manifest.train
        self.eval_partner_agent_lst, self.eval_partner_model_state_lst = _load_partner_agents(eval_entries)
        self.num_train_partners = len(self.frozen_agent_lst)
        self.num_eval_partners = len(self.eval_partner_agent_lst)
        print(f'Partner pool: {self.num_train_partners} training and {len(manifest.heldout)} held-out partner checkpoints '
              f'(see {rundir.path / "partners.yaml"}).')

        if config.ppo.per_agent_num_envs < self.num_train_partners:
            raise ValueError(f'ppo.per_agent_num_envs={config.ppo.per_agent_num_envs} must be >= the number of training '
                             f'partner checkpoints ({self.num_train_partners}) so every partner gets an environment.')
        if config.general.eval_num_episodes < self.num_eval_partners:
            raise ValueError(f'general.eval_num_episodes={config.general.eval_num_episodes} must be >= the number of '
                             f'evaluation partner checkpoints ({self.num_eval_partners}).')
        self.setup(config, rundir)

    def learner_log_name(self, learner_id: int) -> str:
        return 'cooperator' if self.num_learners == 1 else f'cooperator-{learner_id}'

    @property
    def extra_log_info(self) -> dict[str, float]:
        return {'num_train_partners': self.num_train_partners, 'num_eval_partners': self.num_eval_partners}

    def train_env_agent_to_role(self, num_learners: int) -> list[list[int]]:
        # Env e of each cooperator gets partner e % N, with the cooperator in seat e % num_players and the
        # partner in every other seat. As in the paper runs: when N and num_players share a factor (e.g.
        # an even number of partners in 2-player games) a given partner always meets the cooperator in the
        # same seat.
        env_agent_to_role = []
        for learner in range(num_learners):
            for env_idx in range(self.train_per_agent_num_envs):
                partner_role = num_learners + env_idx % self.num_train_partners
                seats = [partner_role] * self.num_players
                seats[env_idx % self.num_players] = learner
                env_agent_to_role.append(seats)
        return env_agent_to_role

    def init_eval(self) -> None:
        model_state_example = self.model_fn_lst[0].init_model_state(jax.random.key(0))
        coop_agent = ActorWrapperAgent(self.model_fn_lst[0])
        print('jax compilation of the [selfplay / crossplay eval rollout] functions...')
        self.eval_selfplay_rollout = self.compile_eval_rollout(
            HomogeneousBatchedEnv(self.env_fn, self.eval_num_episodes), [coop_agent], [model_state_example])
        # Role 0 = the cooperator, role 1 + j = evaluation partner j; episode e pairs the cooperator (in seat
        # e % num_players) with partner e % N_eval.
        env_agent_to_role = []
        for env_idx in range(self.eval_num_episodes):
            seats = [1 + env_idx % self.num_eval_partners] * self.num_players
            seats[env_idx % self.num_players] = 0
            env_agent_to_role.append(seats)
        crossplay_env = FrozenAssignmentEnv(self.env_fn, self.eval_num_episodes, env_agent_to_role=env_agent_to_role)
        self.eval_crossplay_rollout = self.compile_eval_rollout(
            crossplay_env, [coop_agent] + self.eval_partner_agent_lst, [model_state_example] + self.eval_partner_model_state_lst)

    def evaluate(self, learner_id: int, model_state: PyTreeNode) -> tuple[dict[str, float], float]:
        self.rng_eval, rng = jax.random.split(self.rng_eval)
        (reward_seq,) = self.eval_selfplay_rollout(rng, [model_state])
        selfplay = reward_seq.reshape(self.game_length, self.num_players, self.eval_num_episodes).sum(0).mean(0)
        self.rng_eval, rng = jax.random.split(self.rng_eval)
        reward_seq = self.eval_crossplay_rollout(rng, [model_state] + self.eval_partner_model_state_lst)[0]
        crossplay = reward_seq.reshape(self.game_length, self.eval_num_episodes).sum(0)  # cooperator holds one seat per episode
        info = {}
        for name, returns in (('selfplay-return', selfplay), ('crossplay-return', crossplay)):
            info |= {name: returns.mean().item(), f'{name}-std': returns.std().item(),
                     f'{name}-min': returns.min().item(), f'{name}-max': returns.max().item()}
        # 'best' = highest cross-play return against the evaluation (held-out) partners, as in the paper runs.
        return info, info['crossplay-return']


@hydra.main(version_base=None, config_path='../configs', config_name='cecp')
def main(cfg: DictConfig) -> None:
    resolved = resolve_hydra_config(cfg)
    config = train_config_from_mapping(resolved)
    partners_cfg = from_mapping(PartnersConfig, resolved['partners'])
    rundir = create_rundir(config, resolved)
    print(f'Run directory: {rundir.path}')
    CECPController(config, partners_cfg, rundir).train()


if __name__ == '__main__':
    main()
