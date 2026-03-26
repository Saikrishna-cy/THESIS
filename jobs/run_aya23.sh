#!/bin/bash
#SBATCH --job-name=thesis_aya23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/aya23_%j.out
#SBATCH --error=logs/aya23_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4238206@vuw.leidenuniv.nl

# ── Aya-23-8B on ALICE A100 ────────────────────────────────────────────────────
# Cohere multilingual model (23 languages including Dutch)
# HuggingFace ID: CohereForAI/aya-23-8B
# Memory: ~16GB float16 on A100 40GB
# Expected time: ~10-12 hours for full dataset
# Note: Aya-23 may require accepting Cohere's model license on HuggingFace
#       → Visit https://huggingface.co/CohereForAI/aya-23-8B and accept terms
#       → Then: huggingface-cli login --token YOUR_HF_TOKEN
#
# Submit: sbatch jobs/run_aya23.sh
# Monitor: squeue -u $USER
# Logs: tail -f logs/aya23_<JOBID>.out

set -e

PROJECT_DIR="$HOME/THESIS-Hallucination_thesis/THESIS-Hallucination_thesis"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Aya-23-8B Hallucination Pipeline"
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

# Install/verify dependencies
pip install -q transformers accelerate bitsandbytes sentencepiece protobuf

# Cache HuggingFace models in scratch (faster I/O than home directory)
export HF_HOME="$TMPDIR/hf_cache"
export TRANSFORMERS_CACHE="$TMPDIR/hf_cache"

# HuggingFace token for gated models (add your token to .env as HF_TOKEN=hf_...)
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

echo ""
echo "Step 1: Generate RAG responses (Aya-23)"
python experiments/01_pipeline/run_all.py \
    --models aya23 \
    --steps generate \
    --selfcheck-samples 5

echo ""
echo "Step 2: RQ1 — Hallucination taxonomy"
python experiments/01_pipeline/run_all.py \
    --models aya23 \
    --steps rq1

echo ""
echo "Step 3: RQ2 — Detector evaluation"
python experiments/01_pipeline/run_all.py \
    --models aya23 \
    --steps rq2

echo ""
echo "=========================================="
echo "Aya-23 pipeline complete: $(date)"
echo "Results: $PROJECT_DIR/results/aya23/"
echo "=========================================="
