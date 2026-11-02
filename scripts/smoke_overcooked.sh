#!/usr/bin/env bash
# Overcooked smoke run (tiny settings, configs/smoke/oc.yaml): CEC population -> CECP with and without
# next-state prediction, then every Overcooked eval and plot on those runs. Needs a GPU in practice --
# Overcooked is compile-bound on CPU. The numbers are meaningless; this only checks that everything runs.
#   bash scripts/smoke_overcooked.sh [runs_dir] [figures_dir]
set -euo pipefail
RUNS_DIR="${1:-runs/smoke}"
FIG_DIR="${2:-$RUNS_DIR/figures}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUNS_DIR/.jax_cache}"
CEC="$RUNS_DIR/oc_cec" CECP="$RUNS_DIR/oc_cecp" NO_PRED="$RUNS_DIR/oc_cecp_no_pred"

# Training
python "$HERE/train_cec.py"  experiment=oc_cec          smoke=oc logging.runs_dir="$RUNS_DIR"
python "$HERE/train_cecp.py" experiment=oc_cecp         smoke=oc logging.runs_dir="$RUNS_DIR"
python "$HERE/train_cecp.py" experiment=oc_cecp_no_pred smoke=oc logging.runs_dir="$RUNS_DIR"

# Evals (overwrite=true so the script can be re-run on the same runs dir)
EVAL="$HERE/eval" PLOT="$HERE/plot"
TINY_POP="episodes_per_source=2 episode_length=32 replay_batch_size=2 overwrite=true"
python "$EVAL/evaluate_cross_play.py" "runs=[$CEC,$CECP,$NO_PRED]" num_episodes=4 overwrite=true
python "$EVAL/evaluate_skill_decoding.py" run="$CECP" num_episodes_per_partner=10 episode_length=16 overwrite=true
python "$EVAL/evaluate_skill_decoding.py" run="$CEC" latent_key=rnn_hidden partners.num_checkpoints_per_agent=3 \
  num_episodes_per_partner=10 episode_length=16 overwrite=true
python "$EVAL/evaluate_action_prediction.py" run="$CECP" num_episodes_per_partner=4 episode_length=16 overwrite=true
python "$EVAL/visualize_predictions.py" run="$CECP" num_steps=3 overwrite=true
python "$EVAL/evaluate_population_diversity.py" "runs=[$CEC]" "layouts=[cramped_room_9]" $TINY_POP
python "$EVAL/evaluate_cell_action_distributions.py" run="$CEC" "cells=[[1,3]]" inventory=empty \
  min_states_per_orientation=1 $TINY_POP
python "$EVAL/evaluate_partner_stage_mi.py" "runs=[$CEC,$CECP]" "layouts=[cramped_room_9]" max_ego_agents=1 \
  episodes_per_partner=32 episode_length=12 min_timestep=0 min_samples_per_cell=1 n_permutations=8 \
  permutation_chunk=4 overwrite=true
python "$EVAL/visualize_partner_stage_mi_states.py" run="$CECP" layout=cramped_room_9 agent_id=0 top_k=2 overwrite=true

# Plots
python "$PLOT/plot_cross_play.py" "$CEC" "$CECP" "$NO_PRED" --out "$FIG_DIR/cross_play"
python "$PLOT/plot_skill_decoding.py" "$CECP" "$CEC" --out "$FIG_DIR/skill_decoding"
python "$PLOT/plot_action_prediction.py" "$CECP" --out "$FIG_DIR/action_prediction"
python "$PLOT/plot_training_curves.py" "$CECP" "$NO_PRED" --out "$FIG_DIR/training_curves"
python "$PLOT/plot_partner_stage_mi_bar.py" "$CEC" "$CECP" --out "$FIG_DIR/partner_stage_mi"
python "$PLOT/render_overcooked_layouts.py" --out "$FIG_DIR/layouts"
echo "Overcooked smoke runs written to $RUNS_DIR, figures to $FIG_DIR"
