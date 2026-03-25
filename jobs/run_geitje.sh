#!/bin/bash
#SBATCH --job-name=thesis_geitje
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/geitje_%j.out
#SBATCH --error=logs/geitje_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4238206@vuw.leidenuniv.nl

# ── GEITje-7B-Ultra on ALICE A100 ─────────────────────────────────────────────
# Dutch-specific Mistral fine-tune by BramVanroy
# HuggingFace ID: BramVanroy/GEITje-7B-ultra
# Memory: ~14GB float16 on A100 40GB
# Expected time: ~8-10 hours for full dataset (all interviews)
#
# Submit: sbatch jobs/run_geitje.sh
# Monitor: squeue -u $USER
# Logs: tail -f logs/geitje_<JOBID>.out

set -e

PROJECT_DIR="$HOME/THESIS-Hallucination_thesis/THESIS-Hallucination_thesis"
cd "$PROJECT_DIR"

echo "=========================================="
echo "GEITje-7B-Ultra Hallucination Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=========================================="

# Load ALICE modules
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load OpenSSL/1.1

# Activate virtual environment (create once with: python -m venv venv)
source ~/thesis_venv/bin/activate

# Install/verify dependencies
pip install -q transformers accelerate bitsandbytes sentencepiece protobuf

# Cache HuggingFace models in scratch (faster than home on ALICE)
export HF_HOME="$TMPDIR/hf_cache"
export TRANSFORMERS_CACHE="$TMPDIR/hf_cache"

# Load API key for embeddings (needed even for HF generation)
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

echo ""
echo "Step 1: Generate RAG responses (GEITje)"
python experiments/01_pipeline/run_all.py \
    --models geitje \
    --steps generate \
    --selfcheck-samples 5

echo ""
echo "Step 2: RQ1 — Hallucination taxonomy"
python experiments/01_pipeline/run_all.py \
    --models geitje \
    --steps rq1

echo ""
echo "Step 3: RQ2 — Detector evaluation"
python experiments/01_pipeline/run_all.py \
    --models geitje \
    --steps rq2

echo ""
echo "=========================================="
echo "GEITje pipeline complete: $(date)"
echo "Results: $PROJECT_DIR/results/geitje/"
echo "=========================================="
