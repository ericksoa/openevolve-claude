#!/bin/bash
# Run the full ML training pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_DIR/rust"
ML_DIR="$SCRIPT_DIR"

MAX_N=${1:-50}
NUM_RUNS=${2:-200}
EPOCHS=${3:-100}

echo "=== ML Value Function Training Pipeline ==="
echo "MAX_N: $MAX_N, NUM_RUNS: $NUM_RUNS, EPOCHS: $EPOCHS"
echo ""

# Step 1: Build Rust data generator
echo "Step 1: Building Rust data generator..."
cd "$RUST_DIR"
cargo build --release --bin generate_training_data 2>&1 | tail -3

# Step 2: Generate training data
echo ""
echo "Step 2: Generating training data..."
./target/release/generate_training_data $MAX_N $NUM_RUNS "$ML_DIR/training_data.jsonl"

# Step 3: Activate venv and train
echo ""
echo "Step 3: Training model..."
cd "$PROJECT_DIR"
source ml_env/bin/activate

cd "$ML_DIR"
python train.py \
    --data training_data.jsonl \
    --epochs $EPOCHS \
    --max-n $MAX_N \
    --model-type mlp \
    --output value_model.pt

# Step 4: Test beam search
echo ""
echo "Step 4: Testing ML-guided beam search..."
python beam_search.py \
    --model value_model.pt \
    --max-n $MAX_N \
    --beam-width 20 \
    --test-n 20

echo ""
echo "=== Pipeline complete ==="
