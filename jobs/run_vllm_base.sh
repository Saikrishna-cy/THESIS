#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err
#SBATCH --mail-type=END,FAIL

# ── ALICE A100-80GB: vLLM inference (generic template) ───────────────────────
# Usage: sbatch --export=MODEL=geitje jobs/run_vllm_base.sh
#        sbatch --export=MODEL=aya23  jobs/run_vllm_base.sh
#
# vLLM provides 2-24x throughput vs Transformers (Kwon et al. 2023, arXiv:2309.06180)
# Uses PagedAttention for memory efficiency. Reduces ALICE 12h jobs → 2-4h.
#
# MODEL env var: one of gpt-4o-mini, mistral, qwen, qwen14b, geitje, aya23, mixtral
# Default: geitje

MODEL="${MODEL:-geitje}"

echo "============================="
echo "Job: vLLM Inference — $MODEL"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "vLLM: $(pip show vllm 2>/dev/null | grep Version || echo 'not installed')"
echo "============================="

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/thesis_env/bin/activate

PROJECT_ROOT="$HOME/Downloads/THESIS"
cd "$PROJECT_ROOT"
mkdir -p logs "results/$MODEL"

# Run with vLLM backend (USE_VLLM=1 set automatically by --inference vllm)
python experiments/01_pipeline/run_all.py \
    --models "$MODEL" \
    --steps generate,rq1,rq2 \
    --results-base results \
    --inference vllm

echo "============================="
echo "vLLM job COMPLETE — $MODEL"
echo "============================="
