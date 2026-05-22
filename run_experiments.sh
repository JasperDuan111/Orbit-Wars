#!/bin/bash
# Orbit-Wars training launch script
# Runs MLP and GNN models on both CPU and GPU for comparison
# Usage: bash run_experiments.sh

set -e

# Navigate to project root (where rl/ is located)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Common scaled-down parameters
TOTAL_UPDATES=100
EPOCHS=3
BATCH_SIZE=32
NUM_ENVS=4
ROLLOUT_STEPS=64
SAVE_EVERY=25
OPPONENT_REFRESH=5
LEARNING_RATE=3e-4

CHECKPOINT_DIR="checkpoints"
MODEL_DIR="models"

# Create output directories
mkdir -p "$MODEL_DIR"

echo "=============================================="
echo "  Orbit-Wars Training Experiments"
echo "  total_updates=$TOTAL_UPDATES  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE  num_envs=$NUM_ENVS"
echo "=============================================="
echo ""

run_experiment() {
    local MODEL_TYPE=$1
    local DEVICE=$2
    local SAVE_FINAL=$3  # "yes" to save final model
    local DEVICE_FLAG=""
    local FINAL_FLAG=""

    if [ "$DEVICE" == "cpu" ]; then
        DEVICE_FLAG="--device cpu"
    fi
    if [ "$SAVE_FINAL" == "yes" ]; then
        FINAL_FLAG="--save-final-dir $MODEL_DIR --cleanup-checkpoints"
    fi

    local RUN_NAME="${MODEL_TYPE}_${DEVICE}"
    local LOG_PREFIX="[${RUN_NAME}]"

    echo "$LOG_PREFIX Starting training..."
    echo "$LOG_PREFIX   model_type=$MODEL_TYPE  device=$DEVICE  save_final=$SAVE_FINAL"

    python -m rl.train \
        --model-type "$MODEL_TYPE" \
        --total-updates "$TOTAL_UPDATES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-envs "$NUM_ENVS" \
        --rollout-steps "$ROLLOUT_STEPS" \
        --save-every "$SAVE_EVERY" \
        --opponent-refresh "$OPPONENT_REFRESH" \
        --learning-rate "$LEARNING_RATE" \
        --save-dir "$CHECKPOINT_DIR" \
        $DEVICE_FLAG \
        $FINAL_FLAG

    echo "$LOG_PREFIX Done."
    echo ""
}

Run all 4 experiments (only GPU runs save final models)
echo "--- Experiment 1/4: MLP on CPU ---"
run_experiment "mlp" "cpu" "no"

echo "--- Experiment 2/4: MLP on GPU ---"
run_experiment "mlp" "cuda" "yes"

echo "--- Experiment 3/4: GNN on CPU ---"
run_experiment "gnn" "cpu" "no"

echo "--- Experiment 4/4: GNN on GPU ---"
run_experiment "gnn" "cuda" "yes"

echo "=============================================="
echo "  All experiments complete."
echo "  Final models saved in $MODEL_DIR/:"
ls -la "$MODEL_DIR"/
echo "=============================================="
