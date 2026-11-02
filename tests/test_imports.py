import subprocess
import sys

MODULES = [
    'cecp', 'cecp.config', 'cecp.runs', 'cecp.partners', 'cecp.utils',
    'cecp.envs', 'cecp.envs.base', 'cecp.envs.spaces', 'cecp.envs.batched', 'cecp.envs.make',
    'cecp.envs.coop_foraging', 'cecp.envs.overcooked', 'cecp.envs.overcooked.obs_adapter',
    'cecp.models.action_distribution', 'cecp.models.feature_extractors', 'cecp.models.nn_configs',
    'cecp.models.next_state_prediction', 'cecp.models.nn_blocks',
    'cecp.policy.actor_critic', 'cecp.policy.agent_wrapper', 'cecp.policy.build',
    'cecp.trainers.ppo', 'cecp.controller.base', 'cecp.controller.ippo', 'cecp.training',
    'cecp.eval.common', 'cecp.eval.display', 'cecp.eval.rollouts', 'cecp.eval.plotting', 'cecp.eval.overcooked',
    'cecp.eval.diversity',
]


def test_imports_do_not_need_jaxmarl_or_wandb():
    # Fresh interpreter so other tests cannot have imported anything first.
    code = (
        'import importlib, sys\n'
        f'for m in {MODULES!r}: importlib.import_module(m)\n'
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('jaxmarl', 'wandb', 'jax_pbt'))\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)
