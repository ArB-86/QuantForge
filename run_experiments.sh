#!/bin/bash
set -e

CONFIG="configs/lightgbm_regressor.json"

# Build feature cache once
echo "Building feature cache..."
python -m quantforge.cli.main build-features --config $CONFIG

# Run training experiments (baseline, v5) sequentially or in parallel on separate GPUs
TRAIN_EXPS=("baseline" "v5")
PORTFOLIO_EXPS=("hold10" "hold20" "top10" "top20" "equal_weight" "risk_parity")

# Launch training experiments on GPU 0 and 1
for i in "${!TRAIN_EXPS[@]}"; do
    EXP=${TRAIN_EXPS[$i]}
    GPU=$i
    LOG="logs_${EXP}.log"
    echo "Training $EXP on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU \
    python -m quantforge.cli.main experiment \
        --config $CONFIG \
        --experiment $EXP \
        > $LOG 2>&1 &
done

wait
echo "Training experiments completed."

# Now portfolio experiments can reuse predictions
# Launch all portfolio experiments in parallel on remaining GPUs (2-7)
GPUS=(2 3 4 5 6 7)
for i in "${!PORTFOLIO_EXPS[@]}"; do
    EXP=${PORTFOLIO_EXPS[$i]}
    GPU=${GPUS[$i % ${#GPUS[@]}]}
    LOG="logs_${EXP}.log"
    echo "Portfolio $EXP on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU \
    python -m quantforge.cli.main experiment \
        --config $CONFIG \
        --experiment $EXP \
        > $LOG 2>&1 &
done

wait
echo "All experiments completed."
