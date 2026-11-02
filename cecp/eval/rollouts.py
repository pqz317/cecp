"""Compiled rollout functions shared by the evaluation scripts."""
import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from ..controller.base import BaseController
from ..policy.actor_critic import ActorWrapperAgent
from ..utils import split_rng_to_list


def compile_pair_evaluator(
    eval_env: Any,
    episode_length: int,
    env_const: Any,
    model_fn_i: Any,
    model_fn_j: Any,
    model_state_i_example: PyTreeNode,
    model_state_j_example: PyTreeNode,
) -> Callable[[jax.Array, PyTreeNode, PyTreeNode], tuple[jax.Array, jax.Array]]:
    """AOT-compile `(rng, model_state_i, model_state_j) -> (returns_i, returns_j)`.

    `eval_env` is a two-role batched env (role 0 = policy i, role 1 = policy j). Plays one episode of
    `episode_length` steps in every env copy and returns each role's summed reward per sample.
    """
    agents = [ActorWrapperAgent(model_fn_i), ActorWrapperAgent(model_fn_j)]
    batch_sizes = eval_env.get_agent_batch_size()

    def eval_pair(rng: jax.Array, model_state_i: PyTreeNode, model_state_j: PyTreeNode) -> tuple[jax.Array, jax.Array]:
        model_states = [model_state_i, model_state_j]
        rng, rng_init_lst = split_rng_to_list(rng, len(agents))
        agent_states = [agent.init_agent_state(r, b) for r, agent, b in zip(rng_init_lst, agents, batch_sizes)]
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = eval_env.reset(rng_reset, env_const)

        def step(carry, _):
            rng, agent_states, env_state, obs_lst, ret_i, ret_j = carry
            rng, agent_states, env_state, obs_lst, _, reward_lst, _, _ = BaseController.rollout_step(
                env_fn=eval_env, agent_lst=agents, rng=rng, model_state_lst=model_states,
                agent_state_lst=agent_states, env_const=env_const, env_state=env_state, obs_lst=obs_lst,
            )
            return (rng, agent_states, env_state, obs_lst, ret_i + reward_lst[0], ret_j + reward_lst[1]), None

        init = (rng, agent_states, env_state, obs_lst,
                jnp.zeros((batch_sizes[0],), jnp.float32), jnp.zeros((batch_sizes[1],), jnp.float32))
        (_, _, _, _, ret_i, ret_j), _ = jax.lax.scan(step, init, None, length=episode_length)
        return ret_i, ret_j

    print('Compiling pair evaluator...')
    start = time.perf_counter()
    compiled = jax.jit(eval_pair).lower(jax.random.key(0), model_state_i_example, model_state_j_example).compile()
    print(f'Compiled in {time.perf_counter() - start:.1f}s')
    return compiled


@dataclass
class EgoStep:
    """What a per-step output function sees at each step of `make_ego_partner_collector`."""
    rng_forward: jax.Array          # the key the ego's forward pass used
    ego_model_state: PyTreeNode
    ego_agent_state: PyTreeNode     # ego recurrent state before this step
    ego_obs: Any
    ego_next_agent_state: PyTreeNode
    ego_info: dict
    ego_action: Any
    partner_action: Any


def make_ego_partner_collector(
    eval_env: Any,
    episode_length: int,
    env_const: Any,
    ego_model_fn: Any,
    partner_agent: Any,
    step_output: Callable[[EgoStep], Any],
    ego_role: int = 0,
) -> Callable[[jax.Array, PyTreeNode, PyTreeNode], tuple[Any, jax.Array]]:
    """Build `(rng, ego_model_state, partner_model_state) -> (outputs, ego_returns)` (not compiled).

    Plays one episode of `episode_length` steps in every copy of the two-role `eval_env`, with the ego
    model in role `ego_role` and `partner_agent` in the other. `step_output(EgoStep)` picks what to
    record at each step; `outputs` stacks it over time (leading axis T). The ego is run through
    `forward_with_aux`, so step outputs can use its auxiliary info (e.g. `readout`).
    """
    if ego_role not in (0, 1):
        raise ValueError(f'ego_role must be 0 or 1, got {ego_role}')
    partner_role = 1 - ego_role
    agents = [None, None]
    agents[ego_role], agents[partner_role] = ActorWrapperAgent(ego_model_fn), partner_agent
    batch_sizes = eval_env.get_agent_batch_size()

    def collect(rng: jax.Array, ego_model_state: PyTreeNode, partner_model_state: PyTreeNode):
        rng, rng_init_0 = jax.random.split(rng)
        rng, rng_init_1 = jax.random.split(rng)
        agent_states = [agents[0].init_agent_state(rng_init_0, batch_sizes[0]),
                        agents[1].init_agent_state(rng_init_1, batch_sizes[1])]
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = eval_env.reset(rng_reset, env_const)

        def step(carry, _):
            rng, agent_states, env_state, obs_lst, ego_returns = carry
            keys = jax.random.split(rng, 8)   # [next, forward, sample, partner, env, reset_0, reset_1, unused]
            ego_next, ego_pi, _, ego_info = ego_model_fn.forward_with_aux(
                keys[1], ego_model_state, agent_states[ego_role], obs_lst[ego_role])
            ego_action = ego_pi.sample(seed=keys[2])
            partner_next, partner_action, _ = agents[partner_role].step(
                keys[3], partner_model_state, agent_states[partner_role], obs_lst[partner_role])
            output = step_output(EgoStep(keys[1], ego_model_state, agent_states[ego_role], obs_lst[ego_role],
                                         ego_next, ego_info, ego_action, partner_action))
            actions = [None, None]
            actions[ego_role], actions[partner_role] = ego_action, partner_action
            next_env_state, next_obs_lst, reward_lst, done_lst, _ = eval_env.step(keys[4], env_const, env_state, actions)
            next_states = [None, None]
            next_states[ego_role], next_states[partner_role] = ego_next, partner_next
            next_states = [agents[i].reset_agent_state(keys[5 + i], next_states[i], done_lst[i]) for i in (0, 1)]
            return (keys[0], next_states, next_env_state, next_obs_lst, ego_returns + reward_lst[ego_role]), output

        init = (rng, agent_states, env_state, obs_lst, jnp.zeros((batch_sizes[ego_role],), jnp.float32))
        (_, _, _, _, ego_returns), outputs = jax.lax.scan(step, init, None, length=episode_length)
        return outputs, ego_returns

    return collect


def aot_compile(fn: Callable, *example_args: Any, name: str = 'rollout') -> Callable:
    """jit + lower + compile `fn` for the shapes of `example_args`, with a timing printout."""
    print(f'Compiling {name}...')
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*example_args).compile()
    print(f'Compiled in {time.perf_counter() - start:.1f}s')
    return compiled


def make_selfplay_collector(env: Any, episode_length: int, num_episodes: int, model_fn: Any) -> Callable:
    """Build `(rng, model_state, init_state, init_obs) -> (obs, probs, pos, inv, dir, done, returns)` for
    `num_episodes` self-play episodes of one policy in both seats of the unbatched two-player Overcooked
    `env`, all starting from the given fixed state (not compiled).

    Outputs per episode e, seat k, step t (recorded before the step):
        obs [E, 2, T, H, W, C], probs [E, 2, T, A] (the acting policy's distribution),
        pos [E, 2, T, 2] (row, col), inv / dir [E, 2, T] (held object index, orientation),
        done [E, 2, T], returns [E, 2].
    """
    if list(env.get_agent_batch_size()) != [1, 1]:
        raise ValueError(f'Expected per-seat batch size [1, 1], got {env.get_agent_batch_size()}.')
    if episode_length > env.max_steps:
        raise ValueError(f'episode_length={episode_length} exceeds the env max_steps={env.max_steps}: the env would '
                         'auto-reset into its own layout mid-rollout.')
    env_const = env.default_const
    num_seats = 2

    def collect_one(rng, model_state, init_state, init_obs):
        rng, rng_init_0, rng_init_1 = jax.random.split(rng, 3)
        agent_states = [model_fn.init_actor_rnn_state(rng_init_0, 1), model_fn.init_actor_rnn_state(rng_init_1, 1)]

        def step(carry, _):
            rng, agent_states, env_state, obs_lst, returns = carry
            keys = jax.random.split(rng, 2 + 2 * num_seats)   # [next, env, forward x2, sample x2]
            raw = env_state._state
            record = (jnp.concatenate([obs_lst[k]['grid_2d'] for k in range(num_seats)], axis=0),
                      raw.agent_pos[:, ::-1].astype(jnp.int32),   # (x, y) -> (row, col)
                      raw.agent_inv.astype(jnp.int32), raw.agent_dir_idx.astype(jnp.int32))
            next_states, actions, probs = [], [], []
            for k in range(num_seats):
                next_state, pi = model_fn.get_action_distribution(keys[2 + k], model_state, agent_states[k], obs_lst[k])
                actions.append(pi.sample(seed=keys[2 + num_seats + k]))
                probs.append(pi['all'].probs)
                next_states.append(next_state)
            next_env_state, next_obs_lst, reward_lst, done_lst, _ = env.step(keys[1], env_const, env_state, actions)
            obs, pos, inv, direction = record
            outputs = (obs, jnp.concatenate(probs, axis=0), pos, inv, direction, jnp.concatenate(done_lst, axis=0))
            return (keys[0], next_states, next_env_state, next_obs_lst, returns + jnp.concatenate(reward_lst, axis=0)), outputs

        init = (rng, agent_states, init_state, init_obs, jnp.zeros((num_seats,), jnp.float32))
        (*_, returns), seqs = jax.lax.scan(step, init, None, length=episode_length)
        return (*[jnp.swapaxes(x, 0, 1) for x in seqs], returns)   # [T, 2, ...] -> [2, T, ...]

    def collect(rng, model_state, init_state, init_obs):
        return jax.vmap(collect_one, in_axes=(0, None, None, None))(
            jax.random.split(rng, num_episodes), model_state, init_state, init_obs)

    return collect


def make_population_replay(model_fn: Any) -> Callable:
    """Build `(rng, stacked_model_states, obs_batch) -> probs [N, B, T, A]` (not compiled): every member
    j starts from its own zero carry and re-encodes each whole observation sequence {'grid_2d': [B, T, ...]},
    so probs[j, b, t] = pi_j(. | prefix of sequence b up to t) is well defined for recurrent policies."""
    def replay_one(rng, model_state, obs_batch):
        rnn_state = model_fn.init_actor_rnn_state(rng, batch_size=obs_batch['grid_2d'].shape[0])
        _, pi = model_fn.get_action_distribution(rng, model_state, rnn_state, obs_batch)
        return pi['all'].probs
    return jax.vmap(replay_one, in_axes=(None, 0, None))


@dataclass
class FixedStartStep:
    """What a per-step output function sees in `make_fixed_start_ego_collector`."""
    ego_obs: Any
    ego_probs: jax.Array
    env_state: Any          # before the step
    ego_done: jax.Array


def make_fixed_start_ego_collector(
    eval_env: Any,
    episode_length: int,
    env_const: Any,
    ego_model_fn: Any,
    partner_model_fn: Any,
    step_output: Callable[[FixedStartStep], Any],
    ego_role: int = 0,
) -> Callable:
    """Build `(rng, ego_state, partner_state, init_state, init_obs) -> (outputs, ego_returns)` (not compiled):
    ego vs partner in every copy of the batched `eval_env`, from a given (tiled) start state.

    The step body is shared by evaluate_partner_stage_mi (collection) and
    visualize_partner_stage_mi_states (replay), which only differ in `step_output`; the replay
    reproduces the collected rollout exactly because everything else here is the same.
    """
    partner_role = 1 - ego_role
    agents = [None, None]
    agents[ego_role], agents[partner_role] = ActorWrapperAgent(ego_model_fn), ActorWrapperAgent(partner_model_fn)
    batch_sizes = eval_env.get_agent_batch_size()

    def collect(rng, ego_model_state, partner_model_state, init_state, init_obs):
        rng, rng_init_0, rng_init_1 = jax.random.split(rng, 3)
        agent_states = [agents[0].init_agent_state(rng_init_0, batch_sizes[0]),
                        agents[1].init_agent_state(rng_init_1, batch_sizes[1])]

        def step(carry, _):
            rng, agent_states, env_state, obs_lst, ego_returns = carry
            rng_next, rng_fwd, rng_sample, rng_partner, rng_env, rng_reset_0, rng_reset_1 = jax.random.split(rng, 7)
            ego_next, ego_pi = ego_model_fn.get_action_distribution(rng_fwd, ego_model_state, agent_states[ego_role], obs_lst[ego_role])
            ego_action = ego_pi.sample(seed=rng_sample)
            partner_next, partner_action, _ = agents[partner_role].step(
                rng_partner, partner_model_state, agent_states[partner_role], obs_lst[partner_role])
            actions = [None, None]
            actions[ego_role], actions[partner_role] = ego_action, partner_action
            next_env_state, next_obs_lst, reward_lst, done_lst, _ = eval_env.step(rng_env, env_const, env_state, actions)
            next_states = [None, None]
            next_states[ego_role], next_states[partner_role] = ego_next, partner_next
            next_states = [agents[i].reset_agent_state(r, next_states[i], done_lst[i]) for i, r in ((0, rng_reset_0), (1, rng_reset_1))]
            output = step_output(FixedStartStep(obs_lst[ego_role], ego_pi['all'].probs, env_state, done_lst[ego_role]))
            return (rng_next, next_states, next_env_state, next_obs_lst, ego_returns + reward_lst[ego_role]), output

        init = (rng, agent_states, init_state, init_obs, jnp.zeros((batch_sizes[ego_role],), jnp.float32))
        (*_, ego_returns), outputs = jax.lax.scan(step, init, None, length=episode_length)
        return outputs, ego_returns

    return collect


def compile_fixed_start_pair_evaluator(
    env: Any,
    episode_length: int,
    num_episodes: int,
    model_fn_i: Any,
    model_fn_j: Any,
    model_state_i_example: PyTreeNode,
    model_state_j_example: PyTreeNode,
    initial_state_example: Any,
    initial_obs_example: list,
) -> Callable[[jax.Array, PyTreeNode, PyTreeNode, Any, list], tuple[jax.Array, jax.Array]]:
    """AOT-compile `(rng, state_i, state_j, initial_state, initial_obs) -> (returns_i, returns_j)`.

    Plays `num_episodes` episodes of the unbatched two-player `env` from the given start state (e.g.
    an original Overcooked layout, see `cecp.eval.overcooked.fixed_initial_state`); episodes differ
    only in the policies' and environment's randomness. Policy i plays seat 0.
    """
    agents = [ActorWrapperAgent(model_fn_i), ActorWrapperAgent(model_fn_j)]
    batch_sizes = env.get_agent_batch_size()
    env_const = env.default_const

    def eval_episode(rng, model_state_i, model_state_j, initial_state, initial_obs):
        model_states = [model_state_i, model_state_j]
        rng, rng_init_lst = split_rng_to_list(rng, len(agents))
        agent_states = [agent.init_agent_state(r, b) for r, agent, b in zip(rng_init_lst, agents, batch_sizes)]

        def step(carry, _):
            rng, agent_states, env_state, obs_lst, ret_i, ret_j = carry
            rng, agent_states, env_state, obs_lst, _, reward_lst, _, _ = BaseController.rollout_step(
                env_fn=env, agent_lst=agents, rng=rng, model_state_lst=model_states,
                agent_state_lst=agent_states, env_const=env_const, env_state=env_state, obs_lst=obs_lst,
            )
            return (rng, agent_states, env_state, obs_lst, ret_i + reward_lst[0], ret_j + reward_lst[1]), None

        init = (rng, agent_states, initial_state, initial_obs,
                jnp.zeros((batch_sizes[0],), jnp.float32), jnp.zeros((batch_sizes[1],), jnp.float32))
        (_, _, _, _, ret_i, ret_j), _ = jax.lax.scan(step, init, None, length=episode_length)
        return ret_i[0], ret_j[0]

    def eval_pair(rng, model_state_i, model_state_j, initial_state, initial_obs):
        rngs = jax.random.split(rng, num_episodes)
        return jax.vmap(lambda r: eval_episode(r, model_state_i, model_state_j, initial_state, initial_obs))(rngs)

    print('Compiling fixed-start pair evaluator...')
    start = time.perf_counter()
    compiled = jax.jit(eval_pair).lower(jax.random.key(0), model_state_i_example, model_state_j_example,
                                        initial_state_example, initial_obs_example).compile()
    print(f'Compiled in {time.perf_counter() - start:.1f}s')
    return compiled
