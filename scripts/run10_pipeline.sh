#!/usr/bin/env bash
# run10_pipeline.sh
# ------------------------------------------------------------------
# Full Run 10 pipeline on KS4 data:
#   step 1  train Run 10a on full KS4 dataset (~600 neurons)
#   step 2  evaluate Run 10a → per-neuron test correlations
#   step 3  filter dataset at test corr >= 0.3
#   step 4  train Run 10b on filtered KS4 dataset
#   step 5  evaluate Run 10b
#
# Prereq: preprocessed KS4 .npz must already exist (see DATA_FULL below).
# Generate it with:
#   python ~/thesis/preprocess_tvsd.py \
#       --things-dir ~/THINGS_images/object_images \
#       --output     ~/tvsd_monkeyN_V1_sua_KS4.npz \
#       --use-ks4
#
# Run:
#   bash ~/thesis/run10_pipeline.sh                  # foreground
#   nohup bash ~/thesis/run10_pipeline.sh &          # background, survives logout
# ------------------------------------------------------------------

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────
DATA_FULL="$HOME/tvsd_monkeyN_V1_sua_KS4.npz"
DATA_FILTERED="$HOME/tvsd_monkeyN_V1_sua_KS4_filtered03.npz"
RUN10A="$HOME/monkeyN_V1_run10a_ks4_full"
RUN10B="$HOME/monkeyN_V1_run10b_ks4_filtered"
SRC="$HOME/neuralpredictors_scaffold/src"
LOG="$HOME/run10_pipeline.log"

# ── Setup ───────────────────────────────────────────────────────────
source /local/musam/thesis_env/bin/activate
exec > >(tee -a "$LOG") 2>&1

ts() { date '+[%F %T]'; }

# ── Sanity check ────────────────────────────────────────────────────
if [ ! -f "$DATA_FULL" ]; then
    echo "$(ts) ERROR: $DATA_FULL not found." >&2
    echo "$(ts)        Run preprocess_tvsd.py with --use-ks4 first." >&2
    exit 1
fi

# Skip step 1 if Run 10a was already trained (useful for resuming)
if [ -f "$RUN10A/best.pt" ]; then
    echo "$(ts) [skip step 1] $RUN10A/best.pt already exists"
else
    echo "$(ts) === Step 1: train Run 10a on full KS4 (~600 neurons) ==="
    python "$SRC/train.py" \
        --data "$DATA_FULL" \
        --out  "$RUN10A" \
        --backbone convnext_tiny --pretrained --fine-tune \
        --backbone-lr 1e-5 --readout-lr 1e-3 \
        --gamma-readout 0.05 \
        --lr-scheduler cosine --freeze-readout-lr \
        --optimizer adamw --loss nb \
        --epochs 50 --patience 15
fi

# ── Step 2: evaluate Run 10a ────────────────────────────────────────
echo "$(ts) === Step 2: evaluate Run 10a → per-neuron test correlations ==="
python "$SRC/evaluate_neurons.py" \
    --data       "$DATA_FULL" \
    --checkpoint "$RUN10A/best.pt" \
    --out        "$RUN10A/neuron_analysis" \
    --backbone convnext_tiny --pretrained

# ── Step 3: filter at 0.3 ───────────────────────────────────────────
echo "$(ts) === Step 3: filter neurons at test corr >= 0.3 ==="
python "$HOME/thesis/filter_neurons.py" \
    --data      "$DATA_FULL" \
    --analysis  "$RUN10A/neuron_analysis/neuron_analysis_test.npz" \
    --out       "$DATA_FILTERED" \
    --threshold 0.3

# ── Step 4: train Run 10b ───────────────────────────────────────────
echo "$(ts) === Step 4: train Run 10b on filtered KS4 ==="
python "$SRC/train.py" \
    --data "$DATA_FILTERED" \
    --out  "$RUN10B" \
    --backbone convnext_tiny --pretrained --fine-tune \
    --backbone-lr 1e-5 --readout-lr 1e-3 \
    --gamma-readout 0.05 \
    --lr-scheduler cosine --freeze-readout-lr \
    --optimizer adamw --loss nb \
    --epochs 50 --patience 15

# ── Step 5: evaluate Run 10b ────────────────────────────────────────
echo "$(ts) === Step 5: evaluate Run 10b ==="
python "$SRC/evaluate_neurons.py" \
    --data       "$DATA_FILTERED" \
    --checkpoint "$RUN10B/best.pt" \
    --out        "$RUN10B/neuron_analysis" \
    --backbone convnext_tiny --pretrained

echo "$(ts) === Done. Final results in $RUN10B/neuron_analysis ==="
