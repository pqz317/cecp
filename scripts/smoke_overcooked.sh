#!/usr/bin/env bash
# Overcooked smoke run (tiny settings, configs/smoke/oc.yaml): CEC population -> CECP with and without
# next-state prediction. Needs a GPU in practice -- Overcooked is compile-bound on CPU.
#   bash scripts/smoke_overcooked.sh [runs_dir]
set -euo pipefail
RUNS_DIR="${1:-runs/smoke}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUNS_DIR/.jax_cache}"
python "$HERE/train_cec.py"  experiment=oc_cec          smoke=oc logging.runs_dir="$RUNS_DIR"
python "$HERE/train_cecp.py" experiment=oc_cecp         smoke=oc logging.runs_dir="$RUNS_DIR"
python "$HERE/train_cecp.py" experiment=oc_cecp_no_pred smoke=oc logging.runs_dir="$RUNS_DIR"
echo "Overcooked smoke runs written to $RUNS_DIR"
