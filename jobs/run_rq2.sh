#!/bin/bash
#================================================================
# Job: RQ2 Detection Methods (MiniCheck + SelfCheckGPT need GPU)
# Usage: sbatch jobs/run_rq2.sh
#================================================================
#SBATCH --job-name=thesis_rq2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rq2_%j.out
#SBATCH --error=logs/rq2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOURSTUDENTID@vuw.leidenuniv.nl

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load environment
module load Python/3.11.3-GCCcore-12.3.0
source ~/thesis_venv/bin/activate
cd ~/thesis_hallucination

# Verify GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# RQ2: run all 4 detection methods (SelfCheckGPT, MiniCheck, LettuceDetect, RAGAS)
python src/run_all.py --models gpt-4o-mini,qwen,mistral --steps rq2

echo "Job finished: $(date)"
