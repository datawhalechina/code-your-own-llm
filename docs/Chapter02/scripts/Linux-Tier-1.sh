#!/bin/bash

# The $10 tier of nanochat
# Designed to run end-to-end for $10/3 ~= 3.3 hours on a single GPU GPU
# This is a budget-friendly version for experimentation and learning

# all the setup stuff
unset CONDA_PREFIX
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# Then set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=run10 bash run10.sh`
if [ -z "$WANDB_RUN" ]; then
    # Default run name for $10 tier
    WANDB_RUN=run10
fi

# -----------------------------------------------------------------------------
# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~1B characters of pretraining dataset for tokenizer training
# each data shard is ~250M chars, so we download 1e9 / 250e6 = 4 data shards
# each shard is ~100MB of text (compressed), so this is about ~400MB of data on disk
python -m nanochat.dataset -n 4
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 40 is the right number here
python -m nanochat.dataset -n 40 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~1B characters of data
python -m scripts.tok_train --max_chars=1000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# Documenting my process for determining the hyperparameters for this run10.sh script:
# We want a budget of approx. $10 ~= 3.3 hours of single GPU compute
# 1) For a $10 budget on single GPU, we need a much smaller model. Let's target depth=10
# 2) Determine the device_batch_size that fits:
# With depth=10, we use device_batch_size=8 for a good balance of memory and speed.
# For single GPU training, we'll use a smaller total batch size to keep training fast.
# 3) Calculate data requirements:
# A depth=10 model will be approximately ~88M parameters (rough estimate based on scaling)
# Using Chinchilla scaling: #tokens = 20 * #params = 20 * 88M = 1.76B tokens
# At ~4.8 chars/token, this is 1.76B * 4.8 ~= 8.4B chars
# At 250M chars/shard, this is 8.4B / 250M ~= 34 shards needed for pretraining
# Round up to 40 for safety. At ~100MB/shard, this downloads ~4GB of data to disk.
# 4) The training should take roughly 2-2.5 hours, leaving ~1 hour for midtraining, SFT, and evals.

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d10 model on single GPU
# Note: no torchrun needed for single GPU, just use python directly
python -m scripts.base_train --depth=10 --device_batch_size=8 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
python -m scripts.base_loss
# evaluate the model on CORE tasks
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model (single GPU)
python -m scripts.mid_train --device_batch_size=8 --run=$WANDB_RUN
python -m scripts.chat_eval -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
python -m scripts.chat_sft --run=$WANDB_RUN
python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
python -m scripts.chat_rl --run=$WANDB_RUN
# eval the RL model only on GSM8K
python -m scripts.chat_eval -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate

# -----------------------------------------------------------------------------
# Chat with your model!
# Uncomment one of the following to interact with your trained model:

# CLI chat
# python -m scripts.chat_cli -p "Why is the sky blue?"
# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli

# Web UI (ChatGPT style interface)
python -m scripts.chat_web

echo ""
echo "=========================================="
echo "Training complete! Your $10 nanochat model is ready."
echo "To chat with it via CLI: python -m scripts.chat_cli"
echo "To chat with it via Web UI: python -m scripts.chat_web"
echo "=========================================="

