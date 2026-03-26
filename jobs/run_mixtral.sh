#!/bin/bash
#SBATCH --job-name=mixtral_hallucination
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/mixtral_%j.out
#SBATCH --error=logs/mixtral_%j.err
#SBATCH --mail-type=END,FAIL

# ── ALICE A100-80GB: Mixtral-8x7B-Instruct-v0.1 (4-bit quantized) ───────────
# Model: mistralai/Mixtral-8x7B-Instruct-v0.1 (~24-28GB VRAM at 4-bit)
# Paper: Jiang et al. 2024 (arXiv:2401.04088) — MoE: only 12.9B/46.7B active
# TruthfulQA: 71.4% (Table 3 p.5). Hypothesis: MoE routing may reduce
# ROLE_ATTRIBUTION_DRIFT by separating speaker knowledge in expert networks.
# NOTE: 48GB RAM for MoE routing overhead

echo "============================="
echo "Job: Mixtral-8x7B Hallucination"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================="

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/thesis_env/bin/activate

PROJECT_ROOT="$HOME/Downloads/THESIS"
cd "$PROJECT_ROOT"
mkdir -p logs results/mixtral

python experiments/01_pipeline/run_all.py \
    --models mixtral \
    --steps generate,rq1,rq2 \
    --results-base results \
    --inference transformers

echo "============================="
echo "Mixtral-8x7B job COMPLETE"
echo "============================="
