# Orbit-Wars training launch script (PowerShell)
# Runs MLP and GNN models on both CPU and GPU for comparison
# Usage: .\run_experiments.ps1

$ErrorActionPreference = "Stop"

# Navigate to project root (where rl/ is located)
Set-Location $PSScriptRoot

# Common scaled-down parameters
$TOTAL_UPDATES = 100
$EPOCHS = 3
$BATCH_SIZE = 32
$NUM_ENVS = 4
$ROLLOUT_STEPS = 64
$SAVE_EVERY = 25
$OPPONENT_REFRESH = 5
$LEARNING_RATE = "3e-4"

$CHECKPOINT_DIR = "checkpoints"
$MODEL_DIR = "models"

# Create output directories
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null

Write-Host "=============================================="
Write-Host "  Orbit-Wars Training Experiments"
Write-Host "  total_updates=$TOTAL_UPDATES  epochs=$EPOCHS"
Write-Host "  batch_size=$BATCH_SIZE  num_envs=$NUM_ENVS"
Write-Host "=============================================="
Write-Host ""

function Run-Experiment {
    param(
        [string]$ModelType,
        [string]$Device,
        [string]$SaveFinal
    )

    $RunName = "${ModelType}_${Device}"
    $LogPrefix = "[$RunName]"

    Write-Host "$LogPrefix Starting training..."
    Write-Host "$LogPrefix   model_type=$ModelType  device=$Device  save_final=$SaveFinal"

    $pyArgs = @(
        "-m", "rl.train",
        "--model-type", $ModelType,
        "--total-updates", $TOTAL_UPDATES,
        "--epochs", $EPOCHS,
        "--batch-size", $BATCH_SIZE,
        "--num-envs", $NUM_ENVS,
        "--rollout-steps", $ROLLOUT_STEPS,
        "--save-every", $SAVE_EVERY,
        "--opponent-refresh", $OPPONENT_REFRESH,
        "--learning-rate", $LEARNING_RATE,
        "--save-dir", $CHECKPOINT_DIR,
        "--device", $Device
    )
    if ($SaveFinal -eq "yes") {
        $pyArgs += @("--save-final-dir", $MODEL_DIR, "--cleanup-checkpoints")
    }

    python @pyArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$LogPrefix FAILED with exit code $LASTEXITCODE"
        throw "Training failed for $RunName"
    }

    Write-Host "$LogPrefix Done."
    Write-Host ""
}

# Run all 4 experiments (only GPU runs save final models)
# Write-Host "--- Experiment 1/4: MLP on CPU ---"
# Run-Experiment -ModelType "mlp" -Device "cpu" -SaveFinal "no"

# Write-Host "--- Experiment 2/4: MLP on GPU ---"
# Run-Experiment -ModelType "mlp" -Device "cuda" -SaveFinal "yes"

# Write-Host "--- Experiment 3/4: GNN on CPU ---"
# Run-Experiment -ModelType "gnn" -Device "cpu" -SaveFinal "no"

Write-Host "--- Experiment 4/4: GNN on GPU ---"
Run-Experiment -ModelType "gnn" -Device "cuda" -SaveFinal "yes"

Write-Host "=============================================="
Write-Host "  All experiments complete."
Write-Host "  Final models saved in $MODEL_DIR/:"
Get-ChildItem $MODEL_DIR
Write-Host "=============================================="
