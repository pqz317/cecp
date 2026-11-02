"""Train a CEC population: `general.num_agents` independent self-play agents (paper Sec. 4, Stage 1).

When `env.train_layout_mode=random` every episode is played on a freshly sampled procedurally
generated environment (Cross-Environment Cooperation, Jha et al. 2025); evaluation is self-play on the
held-out goal positions / layouts. The resulting run directory is the partner pool for train_cecp.py.

    python scripts/train_cec.py experiment=cf_cec
    python scripts/train_cec.py experiment=oc_cec general.run_name=my_oc_cec
"""
import hydra
import jax
from flax.struct import PyTreeNode
from omegaconf import DictConfig

from cecp.envs.batched import HomogeneousBatchedEnv
from cecp.policy.actor_critic import ActorWrapperAgent
from cecp.runs import RunDir
from cecp.training import TrainConfig, TrainingController, create_rundir, resolve_hydra_config, train_config_from_mapping


class CECController(TrainingController):
    """Each learner plays every seat of its own environments (pure self-play)."""
    train_log_prefix = 'train'
    eval_log_prefix = 'eval'

    def __init__(self, config: TrainConfig, rundir: RunDir) -> None:
        self.setup(config, rundir)

    def learner_log_name(self, learner_id: int) -> str:
        return f'selfplay_agent#{learner_id}'

    def train_env_agent_to_role(self, num_learners: int) -> list[list[int]]:
        num_envs = self.train_per_agent_num_envs * num_learners
        return [[env_idx % num_learners] * self.num_players for env_idx in range(num_envs)]

    def init_eval(self) -> None:
        eval_env_fn = HomogeneousBatchedEnv(self.env_fn, self.eval_num_episodes)
        model_state_example = self.model_fn_lst[0].init_model_state(jax.random.key(0))
        print('jax compilation of the [selfplay eval rollout] function...')
        self.eval_selfplay_rollout = self.compile_eval_rollout(eval_env_fn, [ActorWrapperAgent(self.model_fn_lst[0])], [model_state_example])

    def evaluate(self, learner_id: int, model_state: PyTreeNode) -> tuple[dict[str, float], float]:
        self.rng_eval, rng = jax.random.split(self.rng_eval)
        (reward_seq,) = self.eval_selfplay_rollout(rng, [model_state])
        # The agent fills every seat: [T, num_players * num_episodes] -> per-episode return averaged over seats.
        episode_return = reward_seq.reshape(self.game_length, self.num_players, self.eval_num_episodes).sum(0).mean(0)
        ret = episode_return.mean().item()
        return {'return': ret}, ret


@hydra.main(version_base=None, config_path='../configs', config_name='cec')
def main(cfg: DictConfig) -> None:
    resolved = resolve_hydra_config(cfg)
    config = train_config_from_mapping(resolved)
    rundir = create_rundir(config, resolved)
    print(f'Run directory: {rundir.path}')
    CECController(config, rundir).train()


if __name__ == '__main__':
    main()
