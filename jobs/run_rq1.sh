#!/bin/bash
#================================================================
# Job: RQ1 Hallucination Annotation (API-based, no GPU needed)
# Usage: sbatch jobs/run_rq1.sh
# Note: Uses CPU partition — saves GPU hours for inference jobs
#================================================================
#SBATCH --job-name=thesis_rq1
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rq1_%j.out
#SBATCH --error=logs/rq1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOURSTUDENTID@vuw.leidenuniv.nl

echo "Job started: $(date)"
echo "Running on node: $(hostname)"

# Load environment
module load Python/3.11.3-GCCcore-12.3.0
source ~/thesis_venv/bin/activate
cd ~/thesis_hallucination

# RQ1: hallucination taxonomy annotation (GPT-4o-mini as evaluator via API)
# Requires OPENAI_API_KEY in environment or .env file
python src/run_all.py --models gpt-4o-mini,qwen,mistral --steps rq1

echo "Job finished: $(date)"
