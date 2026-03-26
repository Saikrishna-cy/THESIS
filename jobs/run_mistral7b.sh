#!/bin/bash
#SBATCH --job-name=mistral7b_hallucination
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/mistral7b_%j.out
#SBATCH --error=logs/mistral7b_%j.err
#SBATCH --mail-type=END,FAIL

# ── ALICE A100-80GB: Mistral-7B-Instruct-v0.3 ───────────────────────────────
# Model: mistralai/Mistral-7B-Instruct-v0.3 (~14GB VRAM in float16)
# Paper: Jiang et al. 2023 (arXiv:2310.06825) — GQA + SWA beats LLaMA-2-13B
# No HuggingFace token required — fully public model

echo "============================="
echo "Job: Mistral-7B Hallucination"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================="

# Load modules (ALICE HPC)
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate virtual environment
source ~/thesis_env/bin/activate

# Project root
PROJECT_ROOT="$HOME/Downloads/THESIS"
cd "$PROJECT_ROOT"
mkdir -p logs results/mistral

# Run pipeline — generate + rq1 + rq2
python experiments/01_pipeline/run_all.py \
    --models mistral \
    --steps generate,rq1,rq2 \
    --results-base results \
    --inference transformers

echo "============================="
echo "Mistral-7B job COMPLETE"
echo "============================="
