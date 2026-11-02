"""CPU smoke tests of the evaluation and plot scripts on the tiny Coop Foraging runs from the
session-scoped `cf_runs_dir` fixture (conftest.py). Overcooked evals run in test_train_smoke_gpu.py."""
import json

import numpy as np
import pytest

from smoke_utils import run_script


def test_cross_play(cf_runs_dir, tmp_path):
    runs = [cf_runs_dir / 'cf_cec', cf_runs_dir / 'cf_cecp']
    run_script('eval/evaluate_cross_play.py', f'runs=[{runs[0]},{runs[1]}]', 'num_episodes=4')
    for run in runs:
        out = run / 'eval' / 'cross_play'
        matrix = np.load(out / 'return_matrix.npy')
        assert matrix.shape == (2, 2) and np.isfinite(matrix).all() and np.allclose(matrix, matrix.T)
        summary = json.loads((out / 'summary.json').read_text())
        assert summary['num_pairs'] == 3 and summary['env_mode'] == 'coop_foraging' and summary['eval_on_reserved']
        assert not summary['original_layouts']
        for name in ('pair_metrics.csv', 'per_episode_returns.csv', 'return_matrix.csv', 'eval_config.yaml'):
            assert (out / name).exists()
    with pytest.raises(AssertionError, match='already exists'):
        run_script('eval/evaluate_cross_play.py', f'runs=[{runs[0]}]', 'num_episodes=4')

    run_script('plot/plot_cross_play.py', *map(str, runs), '--out', str(tmp_path / 'xp'))
    assert (tmp_path / 'xp_heldout.png').exists() and (tmp_path / 'xp_heldout.csv').exists()


def test_env_partner_novelty(cf_runs_dir, tmp_path):
    runs = [cf_runs_dir / 'cf_cec', cf_runs_dir / 'cf_cecp']
    run_script('eval/evaluate_env_partner_novelty.py', f'runs=[{runs[0]},{runs[1]}]', 'num_episodes=4')
    for run in runs:
        out = run / 'eval' / 'env_partner_novelty'
        rows = (out / 'results.csv').read_text().splitlines()
        assert rows[0] == 'partner_novelty,env_novelty,mean,std,n_agents' and len(rows) == 5
        assert (out / 'per_agent_returns.csv').exists()
    summary = json.loads((runs[1] / 'eval' / 'env_partner_novelty' / 'summary.json').read_text())
    assert summary['model_type'] == 'cecp' and summary['num_partners'] == {'familiar': 1, 'novel': 1}

    run_script('plot/plot_env_partner_novelty.py', *map(str, runs), '--out', str(tmp_path / 'nov'))
    assert (tmp_path / 'nov.png').exists()


def test_skill_decoding(cf_runs_dir, tmp_path):
    cecp = cf_runs_dir / 'cf_cecp'
    # Partner pool from cf_cecp's partners.yaml: 2 agents x (best + 3 stages).
    run_script('eval/evaluate_skill_decoding.py', f'run={cecp}', 'agent_ids=[0,1]', 'num_episodes_per_partner=10',
               'episode_length=12')
    for agent_id in (0, 1):
        out = cecp / 'eval' / 'skill_decoding' / f'agent_{agent_id}'
        summary = json.loads((out / 'summary.json').read_text())
        assert summary['num_partners'] == 8 and summary['num_runs_with_stage'] == 60
        assert summary['stage_class_names'] == ['early', 'mid', 'late']
        rows = (out / 'checkpoint_stage_decoding_accuracy_by_timestep.csv').read_text().splitlines()
        assert len(rows) == 13 and rows[0].startswith('timestep,accuracy_all_latent_dims,fold_0_accuracy')
        assert np.load(out / 'latents_by_run.npy').shape == (80, 12, 256)
    # CEC ego with an explicit partner pool and the LSTM state as latent.
    cec = cf_runs_dir / 'cf_cec'
    run_script('eval/evaluate_skill_decoding.py', f'run={cec}', 'latent_key=rnn_hidden', f'partners.runs=[{cec}]',
               'partners.num_checkpoints_per_agent=3', 'num_episodes_per_partner=10', 'episode_length=12', 'save_plots=false')
    assert np.load(cec / 'eval' / 'skill_decoding' / 'agent_0' / 'latents_by_run.npy').shape[-1] == 512

    run_script('plot/plot_skill_decoding.py', str(cecp), str(cec), '--out', str(tmp_path / 'sd'))
    assert (tmp_path / 'sd.png').exists()


def test_training_curves(cf_runs_dir, tmp_path):
    run_script('plot/plot_training_curves.py', str(cf_runs_dir / 'cf_cecp'), '--out', str(tmp_path / 'curves'))
    assert (tmp_path / 'curves.png').exists()
