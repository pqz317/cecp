from abc import ABC
from typing import Any, Sequence

import jax
from flax.struct import PyTreeNode

from ..envs import Observation, Action, Env, EnvState, EnvConst
from ..policy.agent_wrapper import RecurrentAgent
from ..runs import MetricsLogger, RunDir, itemize_dict
from ..utils import split_rng_to_list

class BaseController(ABC):
    """Rollout helpers plus run-directory bookkeeping (config, checkpoints, metrics) for training.

    Subclasses call `init_run` once the run directory is known; `log` then appends to the run's
    metrics.csv and, if enabled, to W&B.
    """

    @staticmethod
    def agent_state_inference_step(
        agent_lst: Sequence[RecurrentAgent],
        rng: jax.Array,
        model_state_lst: Sequence[PyTreeNode],
        agent_state_lst: Sequence[PyTreeNode],
        obs_lst: Sequence[Observation],
        done_lst: Sequence[jax.Array]
    ) -> tuple[
        jax.Array,
        list[PyTreeNode],
    ]:
        """
        Inference on the trajectory to get a sequence of agent states.

        [Input]
            - agent_lst: Sequence of agents interacting with the environment.
            - rng: Random number generator state.
            - model_state_lst: Sequence of model states for each agent.
            - agent_state_lst: Sequence of internal agent states.
            - obs_lst: Sequence of observations for each agent.
            - done_lst: Sequence of done-flags for each agent.

        [Output]
            - rng: Random number generator state after one iteration.
            - next_agent_state_lst: The updated agent states after the step.
        """
        rng, rng_act_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst = [
            agent_fn.step(rng_act, model_state, agent_state, obs)[0]
            for rng_act, agent_fn, model_state, agent_state, obs in zip(rng_act_lst, agent_lst, model_state_lst, agent_state_lst, obs_lst)
        ]
        rng, rng_reset_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst = [
            agent_fn.reset_agent_state(rng_reset, next_agent_state, done)
            for rng_reset, agent_fn, next_agent_state, done in zip(
                rng_reset_lst, agent_lst, next_agent_state_lst, done_lst
            )
        ]
        return rng, next_agent_state_lst

    @staticmethod
    def rollout_step(
        env_fn: Env,
        agent_lst: Sequence[RecurrentAgent],
        rng: jax.Array,
        model_state_lst: Sequence[PyTreeNode],
        agent_state_lst: Sequence[PyTreeNode],
        env_const: EnvConst,
        env_state: EnvState,
        obs_lst: Sequence[Observation],
        is_collect_extra_data: bool = False
    ) -> tuple[
        jax.Array,
        EnvState,
        list[PyTreeNode],
        list[Observation],
        list[Action],
        list[jax.Array],
        list[jax.Array],
        dict
    ]:
        """
        Interaction with the environment for one step.

        [Input]
            - env_fn: Environment function that handles state transitions.
            - agent_lst: Sequence of agents interacting with the environment.
            - rng: Random number generator state.
            - model_state_lst: Sequence of model states for each agent.
            - agent_state_lst: Sequence of internal agent states.
            - env_const: Environment constants/configurations.
            - env_state: Current state of the environment.
            - obs_lst: Sequence of observations for each agent.

        [Output]
            - rng: Random number generator state after one iteration.
            - next_agent_state_lst: The updated agent states after the step.
            - next_env_state: The updated environment state after the step.
            - next_obs_lst: New observations received by agents after the step.
            - action_lst: Actions taken by each agent.
            - reward_lst: Rewards received by each agent.
            - done_lst: Boolean flags indicating whether each agent's episode has ended.
            - info: Additional environment-specific information.
        """
        rng, rng_act_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst, action_lst = [], []
        extra_data_info = {}
        for i, (rng_act, agent_fn, model_state, agent_state, obs) in enumerate(zip(rng_act_lst, agent_lst, model_state_lst, agent_state_lst, obs_lst)):
            next_agent_state, action, extra_data = agent_fn.step(rng_act, model_state, agent_state, obs)
            next_agent_state_lst.append(next_agent_state)
            action_lst.append(action)
            if is_collect_extra_data:
                extra_data_info[f'agent-{i}-extra-data'] = extra_data
        
        rng, rng_env_step = jax.random.split(rng)
        next_env_state, next_obs_lst, reward_lst, done_lst, info = env_fn.step(rng_env_step, env_const, env_state, action_lst)
        rng, rng_reset_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst = [
            agent_fn.reset_agent_state(rng_reset, next_agent_state, done)
            for rng_reset, agent_fn, next_agent_state, done in zip(
                rng_reset_lst, agent_lst, next_agent_state_lst, done_lst
            )
        ]
        return rng, next_agent_state_lst, next_env_state, next_obs_lst, action_lst, reward_lst, done_lst, info | extra_data_info

    def init_run(self, run: RunDir, wandb_cfg: dict[str, Any] | None = None) -> None:
        """Attach this controller to `run`. `wandb_cfg` (kwargs for `wandb.init`) enables W&B logging."""
        self.rundir = run
        self.run_dir = str(run.path)
        self.metrics = MetricsLogger(run, wandb_cfg)

    itemize_dict = staticmethod(itemize_dict)

    def log(
        self,
        info: dict[str, Any],
        step: int | None = None,
        log_prefix: str | None = None,
    ) -> None:
        self.metrics.log(info, step=step, log_prefix=log_prefix)
