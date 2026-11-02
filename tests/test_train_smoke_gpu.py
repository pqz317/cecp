"""GPU smoke tests (skipped without a GPU): the Overcooked CEC -> CECP chain, which is compile-bound on CPU,
and the multi-GPU (pmap) training path when at least 2 GPUs are visible. Run with `pytest -m gpu`."""
import jax
import pytest

from smoke_utils import check_run, run_script


def _num_gpus() -> int:
    try:
        return len(jax.devices('gpu'))
    except RuntimeError:
        return 0


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(_num_gpus() == 0, reason='no GPU visible')]


def test_overcooked_cec_to_cecp(tmp_path):
    common = ['smoke=oc', f'logging.runs_dir={tmp_path}']
    run_script('train_cec.py', 'experiment=oc_cec', *common)
    check_run(tmp_path / 'oc_cec', num_agents=2, min_step_ckpts=3, required_metric_keys=['eval/return_mean'])
    run_script('train_cecp.py', 'experiment=oc_cecp', *common)
    check_run(tmp_path / 'oc_cecp', num_agents=2, min_step_ckpts=3,
              required_metric_keys=['cooperator-0_optimization/pred_loss', 'cooperator_eval/crossplay-return_mean'])
    run_script('train_cecp.py', 'experiment=oc_cecp_no_pred', *common)
    check_run(tmp_path / 'oc_cecp_no_pred', num_agents=2, min_step_ckpts=3, required_metric_keys=['cooperator_eval/crossplay-return_mean'])


@pytest.mark.skipif(_num_gpus() < 2, reason='needs >= 2 GPUs')
def test_multi_gpu_sharding(tmp_path):
    n = _num_gpus()
    common = ['smoke=cf', 'sharding.enable_multi_gpu_training=true', f'general.num_agents={n}', f'logging.runs_dir={tmp_path}']
    run_script('train_cec.py', 'experiment=cf_cec', *common)
    check_run(tmp_path / 'cf_cec', num_agents=n, min_step_ckpts=3, required_metric_keys=['eval/return_mean'])
    run_script('train_cecp.py', 'experiment=cf_cecp', *common)
    check_run(tmp_path / 'cf_cecp', num_agents=n, min_step_ckpts=3, required_metric_keys=['cooperator-0_optimization/pred_loss'])
