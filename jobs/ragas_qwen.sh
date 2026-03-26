#!/bin/bash
#SBATCH --job-name=qwen_ragas
#SBATCH --partition=cpu-zen4
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=/zfsstore/user/s4238206/logs/qwen_ragas_%j.out
#SBATCH --error=/zfsstore/user/s4238206/logs/qwen_ragas_%j.err

module load Python/3.11.3-GCCcore-12.3.0
module load OpenSSL/1.1

source ~/thesis_venv/bin/activate

cd /home/s4238206/THESIS-Hallucination_thesis/THESIS-Hallucination_thesis

python -u experiments/01_pipeline/run_all.py --models qwen --steps rq2
