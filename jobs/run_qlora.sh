#!/bin/bash
#SBATCH --job-name=qlora_finetuning
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/qlora_%j.out
#SBATCH --error=logs/qlora_%j.err
#SBATCH --mail-type=END,FAIL

# ── QLoRA Fine-tuning on ALICE A100-80GB ─────────────────────────────────────
# Usage: sbatch --export=MODEL=geitje jobs/run_qlora.sh
#        sbatch --export=MODEL=aya23  jobs/run_qlora.sh
#
# Implements QLoRA (Dettmers et al. 2023, arXiv:2305.17333):
#   - NF4 4-bit quantization (base model: frozen)
#   - LoRA rank-16 adapters on Q/K/V/O attention layers
#   - Training only on FAITHFUL-labeled examples (Wang et al. 2024, arXiv:2311.08401)
#
# Requires: 02_rq1_annotations.json + 01_rag_responses.json already generated
# Output: results/finetuned/{MODEL}_qlora/lora_adapter/

MODEL="${MODEL:-geitje}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
EPOCHS="${EPOCHS:-3}"

echo "============================="
echo "Job: QLoRA Fine-tuning — $MODEL"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "LoRA rank: $LORA_R | Alpha: $LORA_ALPHA | Epochs: $EPOCHS"
echo "============================="

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/thesis_env/bin/activate

PROJECT_ROOT="$HOME/Downloads/THESIS"
cd "$PROJECT_ROOT"
mkdir -p logs "results/finetuned/${MODEL}_qlora"

python experiments/06_finetuning/run_qlora.py \
    --model "$MODEL" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --epochs "$EPOCHS" \
    --output "results/finetuned/${MODEL}_qlora" \
    --results-base results

echo "============================="
echo "QLoRA job COMPLETE — $MODEL"
echo "============================="
