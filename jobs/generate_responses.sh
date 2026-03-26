#!/bin/bash
#================================================================
# Job: Generate AI responses for ALL 6 models
# HF models (qwen, llama, geitje, aya23) run on GPU.
# API models (gpt-4o-mini, mistral) can run without GPU
# but are included here for a single unified job.
#
# Usage:
#   # All 6 models:
#   sbatch jobs/generate_responses.sh
#
#   # HF models only (GPU job):
#   sbatch --export=ALL,MODELS=qwen,llama,geitje,aya23 jobs/generate_responses.sh
#================================================================
#SBATCH --job-name=thesis_generate_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s4238206@vuw.leidenuniv.nl

set -e

PROJECT_DIR="$HOME/THESIS-main"
cd "$PROJECT_DIR"
mkdir -p logs

echo "=========================================="
echo "Thesis — Generate All Responses"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "=========================================="

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi

# Load ALICE modules
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1 2>/dev/null || true

# Activate virtualenv (create first with: python -m venv venv && pip install -r requirements.txt)
source ~/thesis_venv/bin/activate

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
           print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# HuggingFace cache in scratch for fast I/O
export HF_HOME="$TMPDIR/hf_cache"
export TRANSFORMERS_CACHE="$TMPDIR/hf_cache"

# Load env vars (API keys, HF_TOKEN)
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# HF login for gated models (Aya-23 may require this)
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# Which models to run (override with MODELS env var)
MODELS="${MODELS:-gpt-4o-mini,mistral,qwen,llama,geitje,aya23}"
echo "Models: $MODELS"

# Generate responses (resumes from checkpoint if interrupted)
python experiments/01_pipeline/run_all.py \
    --models "$MODELS" \
    --steps generate \
    --selfcheck-samples 5

echo ""
echo "=========================================="
echo "Generation complete: $(date)"
echo "Results: ls $PROJECT_DIR/results/"
ls "$PROJECT_DIR/results/" 2>/dev/null || echo "(empty)"
echo "=========================================="
