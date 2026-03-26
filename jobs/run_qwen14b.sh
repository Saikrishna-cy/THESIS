#!/bin/bash
#SBATCH --job-name=qwen14b_hallucination
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/qwen14b_%j.out
#SBATCH --error=logs/qwen14b_%j.err
#SBATCH --mail-type=END,FAIL

# ── ALICE A100-80GB: Qwen2.5-14B-Instruct (4-bit quantized) ─────────────────
# Model: Qwen/Qwen2.5-14B-Instruct (~10-11GB VRAM at 4-bit NF4)
# Paper: Hui et al. 2024 (arXiv:2412.15115) Table 2 — MMLU 79.7% vs 74.2% (7B)
# 4-bit: Dettmers et al. 2023 (arXiv:2305.17333) QLoRA quantization
# Scale ablation: does 14B < 7B hallucination? McNemar test will confirm.

echo "============================="
echo "Job: Qwen2.5-14B Hallucination"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================="

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/thesis_env/bin/activate

PROJECT_ROOT="$HOME/Downloads/THESIS"
cd "$PROJECT_ROOT"
mkdir -p logs results/qwen14b

python experiments/01_pipeline/run_all.py \
    --models qwen14b \
    --steps generate,rq1,rq2 \
    --results-base results \
    --inference transformers

echo "============================="
echo "Qwen2.5-14B job COMPLETE"
echo "============================="
