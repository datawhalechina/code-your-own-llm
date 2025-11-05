@echo off
REM The $10 tier of nanochat
REM Designed to run end-to-end for $10/3 ~= 3.3 hours on a single RTX 3090/4090/5090 GPU
REM This is a budget-friendly version for experimentation and learning

REM all the setup stuff
set OMP_NUM_THREADS=1
set NANOCHAT_BASE_DIR=%USERPROFILE%\.cache\nanochat
if not exist "%NANOCHAT_BASE_DIR%" mkdir "%NANOCHAT_BASE_DIR%"

REM -----------------------------------------------------------------------------
REM Python venv setup with uv

REM install uv (if not already installed)
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing uv...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

REM create a .venv local virtual environment (if it doesn't exist)
if not exist ".venv" uv venv

REM install the repo dependencies
uv sync --extra gpu

REM activate venv so that `python` uses the project's venv instead of system python
call .venv\Scripts\activate.bat

REM -----------------------------------------------------------------------------
REM wandb setup
REM Make sure to first log in to wandb, e.g. run:
REM    `wandb login`
REM Then set the WANDB_RUN environment variable when running this script, e.g.:
REM    `set WANDB_RUN=Windows-Tier-1 && Windows-Tier-1.bat`
if "%WANDB_RUN%"=="" (
    REM Default run name for $10 tier
    set WANDB_RUN=run10
)

REM -----------------------------------------------------------------------------
REM Initialize report
python -m nanochat.report reset

REM -----------------------------------------------------------------------------
REM Tokenizer

REM Install Rust / Cargo
where cargo >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing Rust...
    powershell -Command "Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile 'rustup-init.exe'; .\rustup-init.exe -y; Remove-Item rustup-init.exe"
    call "%USERPROFILE%\.cargo\env.bat"
)

REM Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

REM Download the first ~1B characters of pretraining dataset for tokenizer training
REM each data shard is ~250M chars, so we download 1e9 / 250e6 = 4 data shards
REM each shard is ~100MB of text (compressed), so this is about ~400MB of data on disk
python -m nanochat.dataset -n 4

REM Immediately also kick off downloading more shards in the background while tokenizer trains
REM See comment below for why 40 is the right number here
start /B python -m nanochat.dataset -n 40

REM train the tokenizer with vocab size 2**16 = 65536 on ~1B characters of data
python -m scripts.tok_train --max_chars=1000000000

REM evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

REM -----------------------------------------------------------------------------
REM Base model (pretraining)

REM Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
set EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if not exist "%NANOCHAT_BASE_DIR%\eval_bundle" (
    echo Downloading eval_bundle...
    powershell -Command "Invoke-WebRequest -Uri '%EVAL_BUNDLE_URL%' -OutFile 'eval_bundle.zip'"
    powershell -Command "Expand-Archive -Path 'eval_bundle.zip' -DestinationPath '.' -Force"
    del eval_bundle.zip
    move eval_bundle "%NANOCHAT_BASE_DIR%"
)

REM Documenting my process for determining the hyperparameters for this run10.bat script:
REM We want a budget of approx. $10 ~= 3.3 hours of single GPU compute
REM 1) For a $10 budget on single GPU, we need a much smaller model. Let's target depth=10
REM 2) Determine the device_batch_size that fits:
REM With depth=10, we use device_batch_size=8 for a good balance of memory and speed.
REM For single GPU training, we'll use a smaller total batch size to keep training fast.
REM 3) Calculate data requirements:
REM A depth=10 model will be approximately ~88M parameters (rough estimate based on scaling)
REM Using Chinchilla scaling: #tokens = 20 * #params = 20 * 88M = 1.76B tokens
REM At ~4.8 chars/token, this is 1.76B * 4.8 ~= 8.4B chars
REM At 250M chars/shard, this is 8.4B / 250M ~= 34 shards needed for pretraining
REM Round up to 40 for safety. At ~100MB/shard, this downloads ~4GB of data to disk.
REM 4) The training should take roughly 2-2.5 hours, leaving ~1 hour for midtraining, SFT, and evals.

echo Waiting for dataset download to complete...
REM Wait for background download process (Windows doesn't have a direct equivalent to wait $PID)
REM The dataset will be downloaded before training starts
timeout /t 5 /nobreak >nul

REM pretrain the d10 model on single GPU
REM Note: no torchrun needed for single GPU, just use python directly
python -m scripts.base_train --depth=10 --device_batch_size=8 --run=%WANDB_RUN%

REM evaluate the model on a larger chunk of train/val data and draw some samples
python -m scripts.base_loss

REM evaluate the model on CORE tasks
python -m scripts.base_eval

REM -----------------------------------------------------------------------------
REM Midtraining (teach the model conversation special tokens, tool use, multiple choice)

REM download 2.3MB of synthetic identity conversations to impart a personality to nanochat
if not exist "%NANOCHAT_BASE_DIR%\identity_conversations.jsonl" (
    echo Downloading identity conversations...
    powershell -Command "Invoke-WebRequest -Uri 'https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl' -OutFile '%NANOCHAT_BASE_DIR%\identity_conversations.jsonl'"
)

REM run midtraining and eval the model (single GPU)
python -m scripts.mid_train --device_batch_size=8 --run=%WANDB_RUN%
python -m scripts.chat_eval -i mid

REM -----------------------------------------------------------------------------
REM Supervised Finetuning (domain adaptation to each sequence all by itself per row)

REM train sft and re-eval right away (should see a small bump)
python -m scripts.chat_sft --run=%WANDB_RUN%
python -m scripts.chat_eval -i sft

REM -----------------------------------------------------------------------------
REM Reinforcement Learning. Optional, and currently only on GSM8K
REM (optional)

REM run reinforcement learning
python -m scripts.chat_rl --run=%WANDB_RUN%

REM eval the RL model only on GSM8K
python -m scripts.chat_eval -i rl -a GSM8K

REM -----------------------------------------------------------------------------
REM Generate the full report by putting together all the sections
REM report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate

REM -----------------------------------------------------------------------------
REM Chat with your model!
REM Uncomment one of the following to interact with your trained model:

REM CLI chat
REM python -m scripts.chat_cli -p "Why is the sky blue?"
REM chat with the model over CLI! Leave out the -p to chat interactively
REM python -m scripts.chat_cli

REM Web UI (ChatGPT style interface)
python -m scripts.chat_web

echo.
echo ==========================================
echo Training complete! Your $10 nanochat model is ready.
echo To chat with it via CLI: python -m scripts.chat_cli
echo To chat with it via Web UI: python -m scripts.chat_web
echo ==========================================

pause

