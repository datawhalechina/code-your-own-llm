@echo off
REM Showing an example run for exercising some of the code paths on the CPU
REM Run as:
REM Windows-Tier-Free.bat

REM NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Windows PC.
REM Think of this run as educational/fun demo, not something you should expect to work well.

echo Starting nanochat CPU demo run on Windows...

REM all the setup stuff
set OMP_NUM_THREADS=1
if not defined NANOCHAT_BASE_DIR set NANOCHAT_BASE_DIR=%USERPROFILE%\.cache\nanochat
if not exist "%NANOCHAT_BASE_DIR%" mkdir "%NANOCHAT_BASE_DIR%"

REM Check if uv is installed, if not, install it
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing uv...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    uv venv
)

REM Install dependencies
echo Installing dependencies...
uv sync --extra cpu

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set WANDB_RUN if not set
if not defined WANDB_RUN set WANDB_RUN=dummy

REM Check if Rust is installed, if not, install it
where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing Rust...
    echo Please run: winget install Rustlang.Rustup
    echo Or download from: https://rustup.rs/
    echo After installing Rust, please restart this script.
    pause
    exit /b 1
)

REM Build the rustbpe tokenizer
echo Building rustbpe tokenizer...
uv run maturin develop --release --manifest-path rustbpe\Cargo.toml

REM wipe the report
echo Resetting report...
python -m nanochat.report reset

REM train tokenizer on ~1B characters
echo Downloading dataset...
python -m nanochat.dataset -n 4
echo Training tokenizer...
python -m scripts.tok_train --max_chars=1000000000
echo Evaluating tokenizer...
python -m scripts.tok_eval

REM train a very small 4 layer model on the CPU
REM each optimization step processes a single sequence of 1024 tokens
REM we only run 50 steps of optimization (bump this to get better results)
echo Training base model (depth=4, 50 iterations)...
python -m scripts.base_train ^
    --depth=4 ^
    --max_seq_len=1024 ^
    --device_batch_size=1 ^
    --total_batch_size=1024 ^
    --eval_every=50 ^
    --eval_tokens=4096 ^
    --core_metric_every=50 ^
    --core_metric_max_per_task=12 ^
    --sample_every=50 ^
    --num_iterations=50

echo Evaluating base model loss...
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096

echo Evaluating base model...
python -m scripts.base_eval --max-per-task=16

REM midtraining
echo Running midtraining...
python -m scripts.mid_train ^
    --max_seq_len=1024 ^
    --device_batch_size=1 ^
    --eval_every=50 ^
    --eval_tokens=4096 ^
    --total_batch_size=1024 ^
    --num_iterations=100

REM eval results will be terrible, this is just to execute the code paths.
REM note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
echo Evaluating chat model (mid)...
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

REM SFT
echo Running supervised fine-tuning...
python -m scripts.chat_sft ^
    --device_batch_size=1 ^
    --target_examples_per_step=4 ^
    --num_iterations=100 ^
    --eval_steps=4 ^
    --eval_metrics_max_problems=16

REM Chat CLI
REM python -m scripts.chat_cli -p "Why is the sky blue?"

REM Chat Web
REM python -m scripts.chat_web

echo Generating final report...
python -m nanochat.report generate

echo.
echo ==========================================
echo Training complete! Check report.md for results.
echo To chat with your model:
echo   python -m scripts.chat_cli
echo   python -m scripts.chat_web
echo ==========================================
pause

