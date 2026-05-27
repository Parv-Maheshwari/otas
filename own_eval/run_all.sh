#!/bin/bash
# Run OTAS evaluation across all 6 dataset variants × 2 modalities sequentially.
# Logs to /home/ubuntu/code/OTAS/result/eval_log.txt. Idempotent — each run writes
# its own subdir under result/Pred/.
#
# Use `env -u LD_LIBRARY_PATH` so the cu132 torch wheel's bundled cuBLAS wins over
# the system /usr/local/cuda-13.2 lib (see reference_blackwell_torch_wheel memory).

set -euo pipefail

cd /home/ubuntu/code/OTAS

LOG=/home/ubuntu/code/OTAS/result/eval_log.txt
mkdir -p /home/ubuntu/code/OTAS/result
: > "$LOG"

run() {
    echo "=== $* ===" | tee -a "$LOG"
    env -u LD_LIBRARY_PATH .venv/bin/python "$@" 2>&1 | tee -a "$LOG"
}

# Order intentionally smallest-first so failures surface fast.
run own_eval/own_MF.py
run own_eval/own_BASEPROD_condensed.py
run own_eval/own_BASEPROD.py
run own_eval/own_GOD.py
run own_eval/own_CART.py --handheld
run own_eval/own_CART.py
# RELLIS-3D — RGB-only off-road benchmark (no thermal modality). Both rows: bare-prompt
# OTAS as in the paper Table V, AND the SAM2 refinement ablation (not in the paper).
run own_eval/own_RELLIS.py
run own_eval/own_RELLIS.py --enable_mask_refinement

echo "=== ALL DONE ===" | tee -a "$LOG"
