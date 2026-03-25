#!/bin/bash
#SBATCH --job-name=llama_all
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/zfsstore/user/s4238206/logs/llama_all_%j.out
#SBATCH --error=/zfsstore/user/s4238206/logs/llama_all_%j.err

# ── Llama-3.1-8B-Instruct on ALICE A100-80G ───────────────────────────────────
# Meta open model — 8B params, 128K context
# HuggingFace ID: meta-llama/Llama-3.1-8B-Instruct
# Memory: ~16GB float16 on A100 80GB
# Expected time: ~22h generate + ~4h RQ1 + ~10h RQ2 = ~36h total

PROJECT_DIR="$HOME/THESIS-Hallucination_thesis/THESIS-Hallucination_thesis"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Llama-3.1-8B-Instruct Hallucination Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=========================================="

# Load ALICE modules
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load OpenSSL/1.1

# Activate virtual environment
source ~/thesis_venv/bin/activate

# Cache HuggingFace models in scratch (faster I/O)
export HF_HOME="$TMPDIR/hf_cache"
export TRANSFORMERS_CACHE="$TMPDIR/hf_cache"

# Load API keys and HF token
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

echo ""
echo "Step 1: Generate RAG responses (Llama)"
python -u experiments/01_pipeline/run_all.py \
    --models llama \
    --steps generate \
    --selfcheck-samples 5

echo ""
echo "Step 2: RQ1 — Hallucination taxonomy"
python -u experiments/01_pipeline/run_all.py \
    --models llama \
    --steps rq1

echo ""
echo "Step 3: RQ2 — Detector evaluation"
python -u experiments/01_pipeline/run_all.py \
    --models llama \
    --steps rq2

echo ""
echo "=========================================="
echo "Llama pipeline complete: $(date)"
echo "Results: $PROJECT_DIR/results/llama/"
echo "=========================================="
