"""GPU smoke tests (skipped without a GPU): the Overcooked CEC -> CECP chain and the Overcooked evals,
which are compile-bound on CPU, and the multi-GPU (pmap) training path when at least 2 GPUs are
visible. Run with `pytest -m gpu`."""
import json

import jax
import numpy as np
import pytest

from cecp.eval.plotting import read_csv_rows
from smoke_utils import check_run, run_script


def _num_gpus() -> int:
    try:
        return len(jax.devices('gpu'))
    except RuntimeError:
        return 0


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(_num_gpus() == 0, reason='no GPU visible')]


@pytest.fixture(scope='module')
def oc_runs_dir(tmp_path_factory):
    runs_dir = tmp_path_factory.mktemp('oc_runs')
    common = ['smoke=oc', f'logging.runs_dir={runs_dir}']
    run_script('train_cec.py', 'experiment=oc_cec', *common)
    run_script('train_cecp.py', 'experiment=oc_cecp', *common)
    run_script('train_cecp.py', 'experiment=oc_cecp_no_pred', *common)
    return runs_dir


def test_overcooked_cec_to_cecp(oc_runs_dir):
    check_run(oc_runs_dir / 'oc_cec', num_agents=2, min_step_ckpts=3, required_metric_keys=['eval/return_mean'])
    check_run(oc_runs_dir / 'oc_cecp', num_agents=2, min_step_ckpts=3,
              required_metric_keys=['cooperator-0_optimization/pred_loss', 'cooperator_eval/crossplay-return_mean'])
    check_run(oc_runs_dir / 'oc_cecp_no_pred', num_agents=2, min_step_ckpts=3, required_metric_keys=['cooperator_eval/crossplay-return_mean'])


def test_overcooked_cross_play(oc_runs_dir, tmp_path):
    runs = [oc_runs_dir / 'oc_cec', oc_runs_dir / 'oc_cecp']
    run_script('eval/evaluate_cross_play.py', f'runs=[{runs[0]},{runs[1]}]', 'num_episodes=4')
    for run in runs:
        out = run / 'eval' / 'cross_play'
        summary = json.loads((out / 'summary.json').read_text())
        assert summary['env_mode'] == 'overcooked' and len(summary['original_layouts']) == 5
        for name in ['', *summary['original_layouts'], 'original_avg']:
            matrix = np.load(out / f'return_matrix{"_" + name if name else ""}.npy')
            assert matrix.shape == (2, 2) and np.isfinite(matrix).all()
    run_script('plot/plot_cross_play.py', *map(str, runs), '--out', str(tmp_path / 'xp'))
    assert (tmp_path / 'xp_original_layouts.png').exists() and (tmp_path / 'xp_original_layouts_significance.csv').exists()


def test_overcooked_latent_and_prediction_evals(oc_runs_dir, tmp_path):
    cecp, cec = oc_runs_dir / 'oc_cecp', oc_runs_dir / 'oc_cec'
    run_script('eval/evaluate_skill_decoding.py', f'run={cecp}', 'num_episodes_per_partner=10', 'episode_length=16')
    assert (cecp / 'eval' / 'skill_decoding' / 'agent_0' / 'checkpoint_stage_decoding_accuracy_by_timestep.csv').exists()
    # A CEC ego has no partners.yaml, so its partner pool defaults to its own population.
    run_script('eval/evaluate_skill_decoding.py', f'run={cec}', 'latent_key=rnn_hidden', 'partners.num_checkpoints_per_agent=3',
               'num_episodes_per_partner=10', 'episode_length=16')
    run_script('plot/plot_skill_decoding.py', str(cecp), str(cec), '--out', str(tmp_path / 'sd'))

    run_script('eval/evaluate_action_prediction.py', f'run={cecp}', 'num_episodes_per_partner=4', 'episode_length=16')
    summary = json.loads((cecp / 'eval' / 'action_prediction' / 'summary.json').read_text())
    assert summary['num_partners'] == 2 and 0.0 <= summary['overall_accuracy'] <= 1.0
    run_script('plot/plot_action_prediction.py', str(cecp), '--out', str(tmp_path / 'ap'))
    assert (tmp_path / 'ap_bar.png').exists() and (tmp_path / 'ap_by_timestep.png').exists()

    run_script('eval/visualize_predictions.py', f'run={cecp}', 'num_steps=3')
    assert len(list((cecp / 'eval' / 'prediction_frames').glob('step_*.png'))) == 3

    run_script('plot/plot_training_curves.py', str(cecp), str(oc_runs_dir / 'oc_cecp_no_pred'), '--out', str(tmp_path / 'curves'))
    run_script('plot/render_overcooked_layouts.py', '--out', str(tmp_path / 'layouts'))
    assert (tmp_path / 'curves.png').exists() and (tmp_path / 'layouts.pdf').exists()



def test_overcooked_state_analyses(oc_runs_dir, tmp_path):
    cecp, cec = oc_runs_dir / 'oc_cecp', oc_runs_dir / 'oc_cec'
    tiny = ['episodes_per_source=2', 'episode_length=32', 'replay_batch_size=2']
    run_script('eval/evaluate_population_diversity.py', f'runs=[{cec}]', 'layouts=[cramped_room_9]', *tiny)
    row = next(r for r in read_csv_rows(cec / 'eval' / 'population_diversity' / 'table2.csv') if r['layout'] == 'cramped_room_9')
    # Two agents: the population JSD is the single pairwise JSD.
    assert np.isclose(float(row['jsd_pop']), float(row['min_jsd_pair']))

    run_script('eval/evaluate_cell_action_distributions.py', f'run={cec}', 'cells=[[1,3]]', 'inventory=empty',
               'min_states_per_orientation=1', *tiny)
    cell_dir = cec / 'eval' / 'cell_action_distributions' / 'coord_ring_9' / 'cell_r1c3_empty'
    assert (cell_dir / 'action_probs_by_agent.csv').exists() and (cell_dir / 'action_probs_by_agent_matrix.png').exists()

    mi_args = ['layouts=[cramped_room_9]', 'max_ego_agents=1', 'episodes_per_partner=32', 'episode_length=12',
               'min_timestep=0', 'min_samples_per_cell=1', 'n_permutations=8', 'permutation_chunk=4']
    run_script('eval/evaluate_partner_stage_mi.py', f'runs=[{cec},{cecp}]', *mi_args)
    for run in (cec, cecp):
        rows = read_csv_rows(run / 'eval' / 'partner_stage_mi' / 'partner_stage_mi_by_ego.csv')
        assert len(rows) == 1 and int(rows[0]['n_states_qualifying']) > 0
        assert np.load(run / 'eval' / 'partner_stage_mi' / 'cramped_room_9' / 'agent_0' / 'null.npy').shape == (8, 2)
    run_script('eval/visualize_partner_stage_mi_states.py', f'run={cecp}', 'layout=cramped_room_9', 'agent_id=0', 'top_k=2')
    assert len(read_csv_rows(cecp / 'eval' / 'partner_stage_mi_states' / 'cramped_room_9' / 'agent_0' / 'top_states.csv')) == 2
    run_script('plot/plot_partner_stage_mi_bar.py', str(cec), str(cecp), '--out', str(tmp_path / 'mi'))
    assert (tmp_path / 'mi.png').exists() and (tmp_path / 'mi_significance.csv').exists()

@pytest.mark.skipif(_num_gpus() < 2, reason='needs >= 2 GPUs')
def test_multi_gpu_sharding(tmp_path):
    n = _num_gpus()
    common = ['smoke=cf', 'sharding.enable_multi_gpu_training=true', f'general.num_agents={n}', f'logging.runs_dir={tmp_path}']
    run_script('train_cec.py', 'experiment=cf_cec', *common)
    check_run(tmp_path / 'cf_cec', num_agents=n, min_step_ckpts=3, required_metric_keys=['eval/return_mean'])
    run_script('train_cecp.py', 'experiment=cf_cecp', *common)
    check_run(tmp_path / 'cf_cecp', num_agents=n, min_step_ckpts=3, required_metric_keys=['cooperator-0_optimization/pred_loss'])
