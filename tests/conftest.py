import pytest

from cecp.config import ArchitectureConfig, CoopForagingEnvConfig, OvercookedEnvConfig


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
