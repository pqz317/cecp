# CECP: Cross-Environment Co-Play

Code for **Unifying Partner and Environment Diversity to Improve Human-AI Coordination**
(Patrick Q. Zhang, Yancheng Liang, Adrienne Fairhall, Natasha Jaques; NeurIPS 2026).

Zero-shot coordination usually trains agents against diverse partners in a single fixed environment.
Cross-Environment Co-Play (CECP) targets coordination with novel partners in novel environments at the
same time, using a two-stage framework:

1. **Partner population.** Train a population of agents independently via self-play across a
   distribution of procedurally generated environments (Cross-Environment Cooperation, CEC;
   Jha et al. 2025). Each agent's early, middle and late training checkpoints are added to the
   partner population, which diversifies partner skill level.
2. **Cooperator.** Train a cooperator against the frozen partner population, again across the
   distribution of environments. The cooperator has an auxiliary next-state prediction objective: a
   prediction head decodes the next state from the cooperator's latent representation and its own
   action. Predicting the next state requires modeling both the partner's behavior and the
   environment's dynamics.

Everything is written in JAX. There are two environments: Cooperative Foraging, a grid world in which
both agents must reach and stay on the same goal, with held-out goal locations; and Overcooked, with
procedurally generated 9×9 layouts and the five original layouts held out.

## Installation

Python ≥ 3.10. The dependencies are pinned to the JAX 0.4.18 / Flax 0.7.4 stack the paper's models
were trained with.

```bash
pip install -e .                 # add [wandb] for optional W&B logging, [dev] for the tests
# GPU: install the matching CUDA build of jaxlib
pip install "jax[cuda12_pip]==0.4.18" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Overcooked is expensive to compile: expect several minutes of XLA compilation per script, even on a
GPU. Set `JAX_COMPILATION_CACHE_DIR` to reuse compiled graphs across runs.

## Training

The configs use [Hydra](https://hydra.cc). Each paper model is one experiment config in
`configs/experiment/`:

| Experiment | Script | What it trains |
|---|---|---|
| `cf_cec` | `train_cec.py` | Cooperative Foraging CEC population (10 agents) |
| `cf_cecp` | `train_cecp.py` | Cooperative Foraging CECP cooperators, against `runs/cf_cec` |
| `oc_cec` | `train_cec.py` | Overcooked CEC population (8 agents) |
| `oc_cecp` | `train_cecp.py` | Overcooked CECP cooperators, against `runs/oc_cec` |
| `oc_cecp_no_pred` | `train_cecp.py` | Ablation: `oc_cecp` without next-state prediction (Fig 8) |

```bash
python scripts/train_cec.py  experiment=oc_cec
python scripts/train_cecp.py experiment=oc_cecp        # partners: runs/oc_cec
```

- **Runs and overrides.** Every run is 3B environment steps per agent. Any config value can be
  overridden on the command line, e.g. `general.seed=1` or `general.run_name=my_run`. Use
  `--cfg job` to print the resolved config without training.
- **Multiple GPUs.** The paper experiments shard agents across GPUs
  (`sharding.enable_multi_gpu_training=true`), which needs at least 2 visible devices. On a single
  GPU, add `sharding.enable_multi_gpu_training=false`.
- **Partner pool.** CECP runs read the pool from `partners.runs`. By default this is the matching
  CEC run under the same `logging.runs_dir`.
- **Held-out partners.** A fraction (`partners.heldout_ratio`, 0.2) of partner *agents* is held out.
  These partners are used for evaluation and for choosing each cooperator's `best` checkpoint.
- **W&B.** Metrics go to `metrics.csv`. Add `logging.wandb=true` to also log to Weights & Biases.

Each run writes a self-contained directory:

```
runs/<run_name>/
  config.yaml                              # full resolved config
  metrics.csv                              # step,key,value
  partners.yaml                            # CECP only: the partner checkpoints trained against / held out
  agents/agent_<i>/{best,step_<n>}/params.msgpack
  eval/<eval_name>/                        # written by the evaluation scripts
```

Checkpoints hold parameters only, and models are rebuilt from `config.yaml`
(`cecp.runs.load_policy`). This repository does not include trained models.

## Evaluation

The evaluation scripts in `scripts/eval/` take run directories as input. They build the environment
from the run's own `config.yaml` and write their results to `<run>/eval/<eval_name>/`, along with the
`eval_config.yaml` they used. They refuse to overwrite an existing output directory unless you pass
`overwrite=true`. Their default settings are in `configs/eval/<script>.yaml`.

The plot scripts in `scripts/plot/` read those outputs, take plain argparse arguments, and write
`<out>.png` and `<out>.svg`. Each script's docstring describes its outputs.
In the table, CF is Cooperative Foraging and OC is Overcooked.

| Analysis | Env | Commands |
|---|---|---|
| Cross-play return between all agents of a population; Overcooked runs also play the five original layouts | CF, OC | `eval/evaluate_cross_play.py runs=[runs/oc_cec,runs/oc_cecp]` → `plot/plot_cross_play.py runs/oc_cec runs/oc_cecp` |
| Return with familiar vs novel partners, in familiar vs novel environments | CF | `eval/evaluate_env_partner_novelty.py runs=[runs/cf_cec,runs/cf_cecp]` → `plot/plot_env_partner_novelty.py runs/cf_cec runs/cf_cecp` |
| Cooperator return against held-out partners over training | CF, OC | `plot/plot_training_curves.py runs/oc_cecp runs/oc_cecp_no_pred` |
| Decoding the partner's training stage from the cooperator's latent state, with PCA trajectories | CF, OC | `eval/evaluate_skill_decoding.py run=runs/oc_cecp` → `plot/plot_skill_decoding.py runs/oc_cecp` |
| Accuracy of the partner movement implied by the cooperator's next-state prediction | OC | `eval/evaluate_action_prediction.py run=runs/oc_cecp` → `plot/plot_action_prediction.py runs/oc_cecp` |
| Frames of the cooperator's predicted next state vs the true one | OC | `eval/visualize_predictions.py run=runs/oc_cecp` |
| How much the cooperator's action depends on its partner's training stage, I(a; z \| s), vs a permutation null | OC | `eval/evaluate_partner_stage_mi.py runs=[runs/oc_cecp]` → `plot/plot_partner_stage_mi_bar.py runs/oc_cecp` |
| The states where the action depends most on the partner's stage | OC | `eval/visualize_partner_stage_mi_states.py run=runs/oc_cecp` (after `evaluate_partner_stage_mi.py`) |
| Behavioral diversity of a population: entropy gap and pairwise JSD of action distributions | OC | `eval/evaluate_population_diversity.py runs=[runs/oc_cec]` |
| Each population member's action distribution at one grid cell | OC | `eval/evaluate_cell_action_distributions.py run=runs/oc_cec` |
| Renders of the five original Overcooked layouts | OC | `plot/render_overcooked_layouts.py` |

All paths are relative to `scripts/`. Run them as, for example,
`python scripts/eval/evaluate_cross_play.py runs=[runs/oc_cec,runs/oc_cecp]`.

## Tests

```bash
pytest -m "not gpu"    # CPU: unit tests + Cooperative Foraging train → eval → plot smoke chain (up to ~1 h)
pytest -m gpu          # Overcooked chain, every Overcooked eval and plot, multi-GPU training (~1h45 on 2 GPUs)
bash scripts/smoke_overcooked.sh runs/smoke    # the same Overcooked chain outside pytest
```

The smoke tests train tiny models (`configs/smoke/`) and only check that every script runs and writes
its outputs. Almost all of their run time is XLA compilation.

## Citation

```bibtex
@inproceedings{zhang2026unifying,
  title     = {Unifying Partner and Environment Diversity to Improve Human-AI Coordination},
  author    = {Zhang, Patrick Q. and Liang, Yancheng and Fairhall, Adrienne and Jaques, Natasha},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The training core is derived from
[Social-RL/jax_pbt](https://github.com/Social-RL/jax_pbt) (MIT). The Overcooked environment is
vendored from [JaxMARL](https://github.com/FLAIROx/JaxMARL) (Apache-2.0), as extended by
[crossEnvCooperation](https://github.com/pqz317/crossEnvCooperation) (MIT). See
`THIRD_PARTY_NOTICES.md` for details.
