import os

# The GPU smoke tests run training scripts as subprocesses while this process also holds JAX arrays;
# without this, the first JAX process would preallocate most of the GPU memory and starve the others.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import pytest

from cecp.config import ArchitectureConfig, CoopForagingEnvConfig, OvercookedEnvConfig


@pytest.fixture(scope='session')
def cf_runs_dir(tmp_path_factory):
    """Tiny Coop Foraging CEC population and CECP cooperators (configs/smoke/cf.yaml), trained once per
    test session through the real scripts and shared by the training and eval smoke tests."""
    from smoke_utils import run_script

    runs_dir = tmp_path_factory.mktemp('cf_runs')
    run_script('train_cec.py', 'experiment=cf_cec', 'smoke=cf', f'logging.runs_dir={runs_dir}')
    run_script('train_cecp.py', 'experiment=cf_cecp', 'smoke=cf', f'logging.runs_dir={runs_dir}')
    return runs_dir


@pytest.fixture
def cf_env_cfg() -> CoopForagingEnvConfig:
    # Paper Coop Foraging setting: no obstacles, 20% of goal positions held out for evaluation.
    return CoopForagingEnvConfig(obstacle_density=0.0, heldout_goal_ratio=0.2, heldout_goal_seed=42, eval_on_reserved_goals=True)


@pytest.fixture
def oc_env_cfg() -> OvercookedEnvConfig:
    # Small held-out pool keeps the test fast; the paper uses 500 samples.
    return OvercookedEnvConfig(heldout_layout_seed=42, heldout_layout_num_samples=4, eval_on_reserved_layouts=True)


@pytest.fixture
def cf_arch() -> ArchitectureConfig:
    # The paper's Coop Foraging runs used 3-layer actor/critic heads.
    return ArchitectureConfig(actor_head_sizes=[256, 192, 128], critic_head_sizes=[512, 256, 192])


@pytest.fixture
def oc_arch() -> ArchitectureConfig:
    return ArchitectureConfig()
