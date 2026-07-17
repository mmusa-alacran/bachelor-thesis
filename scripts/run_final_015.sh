#!/usr/bin/env bash
# run_final_015.sh
# ------------------------------------------------------------------
# Final thesis run: 0.15 Cadena neuron cut.
#   step 1  filter full KS4 dataset at test corr >= 0.15
#           (using Run 10a's per-neuron test correlations)
#   step 2  train on the 0.15-filtered KS4 dataset with Run 10b hyperparameters
#   step 3  evaluate
#
# Mirrors run10_pipeline.sh but with threshold 0.15 and new output dirs,
# reusing Run 10a's already-computed per-neuron test correlations.
# ------------------------------------------------------------------

set -euo pipefail

# Pin to the free GPU (GPU 0); GPU 1 has another process running.
export CUDA_VISIBLE_DEVICES=0

DATA_FULL="$HOME/tvsd_monkeyN_V1_sua_KS4.npz"
DATA_FILTERED="$HOME/tvsd_monkeyN_V1_sua_KS4_filtered015.npz"
RUN10A="$HOME/monkeyN_V1_run10a_ks4_full"
RUN_FINAL="$HOME/monkeyN_V1_run10c_ks4_filtered015"
SRC="$HOME/neuralpredictors_scaffold/src"
LOG="$HOME/run_final_015.log"

source ... #/venv/bin/activate

ts() { date '+[%F %T]'; }

echo "$(ts) === Step 1: filter neurons at test corr >= 0.15 ===" | tee -a "$LOG"
python "$HOME/thesis/filter_neurons.py" \
    --data      "$DATA_FULL" \
    --analysis  "$RUN10A/neuron_analysis/neuron_analysis_test.npz" \
    --out       "$DATA_FILTERED" \
    --threshold 0.15 2>&1 | tee -a "$LOG"

echo "$(ts) === Step 2: train final model on 0.15-filtered KS4 ===" | tee -a "$LOG"
python "$SRC/train.py" \
    --data "$DATA_FILTERED" \
    --out  "$RUN_FINAL" \
    --backbone convnext_tiny --pretrained --fine-tune \
    --backbone-lr 1e-5 --readout-lr 1e-3 \
    --gamma-readout 0.05 \
    --lr-scheduler cosine --freeze-readout-lr \
    --optimizer adamw --loss nb \
    --epochs 50 --patience 15 2>&1 | tee -a "$LOG"

echo "$(ts) === Step 3: evaluate final model ===" | tee -a "$LOG"
python "$SRC/evaluate_neurons.py" \
    --data       "$DATA_FILTERED" \
    --checkpoint "$RUN_FINAL/best.pt" \
    --out        "$RUN_FINAL/neuron_analysis" \
    --backbone convnext_tiny --pretrained 2>&1 | tee -a "$LOG"

echo "$(ts) === Done. Final results in $RUN_FINAL/neuron_analysis ===" | tee -a "$LOG"
