"""End-to-end CPU smoke test: CEC population -> CECP cooperators (with and without next-state prediction)
on Coop Foraging, through the real training scripts with tiny settings (configs/smoke/cf.yaml). The
CEC and CECP runs come from the session-scoped `cf_runs_dir` fixture (conftest.py), which the eval
smoke tests reuse."""
from smoke_utils import check_run, read_metrics, run_script


def test_cec_population(cf_runs_dir):
    check_run(cf_runs_dir / 'cf_cec', num_agents=2, min_step_ckpts=3,
              required_metric_keys=['selfplay_agent#0_train/avg-return', 'selfplay_agent#1_eval/return',
                                    'selfplay_agent#0_optimization/loss', 'train/avg-return_mean', 'eval/return_mean'])


def test_cecp_with_prediction(cf_runs_dir):
    run = check_run(cf_runs_dir / 'cf_cecp', num_agents=2, min_step_ckpts=3,
                    required_metric_keys=['cooperator-0_train/avg-return', 'cooperator-1_eval/crossplay-return',
                                          'cooperator-0_eval/selfplay-return', 'cooperator-0_optimization/pred_loss',
                                          'cooperator_eval/crossplay-return_mean'])
    manifest = run.read_partners()
    assert manifest.partner_runs == [str(cf_runs_dir / 'cf_cec')]
    # 2 partner agents, heldout_ratio 0.5 -> one held out; each contributes best + 3 step checkpoints.
    assert {e.agent_id for e in manifest.train}.isdisjoint({e.agent_id for e in manifest.heldout})
    assert len(manifest.train) == 4 and len(manifest.heldout) == 4
    assert read_metrics(run)['cooperator_train/num_eval_partners'][0] == 4


def test_cecp_without_prediction(cf_runs_dir):
    run_script('train_cecp.py', 'experiment=cf_cecp', 'smoke=cf', 'predictive.use_predictive_objective=false',
               'general.run_name=cf_cecp_no_pred', 'general.per_agent_env_steps=1440', f'logging.runs_dir={cf_runs_dir}')
    run = check_run(cf_runs_dir / 'cf_cecp_no_pred', num_agents=2, min_step_ckpts=3,
                    required_metric_keys=['cooperator-0_optimization/loss'])
    assert 'cooperator-0_optimization/pred_loss' not in read_metrics(run)
