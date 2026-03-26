#!/bin/bash
#SBATCH --job-name=bge_m3_embeddings
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/embeddings_%j.out
#SBATCH --error=logs/embeddings_%j.err
#SBATCH --mail-type=END,FAIL

# ── Pre-compute BGE-M3 embeddings for all transcripts ────────────────────────
# Model: BAAI/bge-m3 — unified embedding model for multilingual retrieval
# Paper: Chen et al. 2024 (arXiv:2402.03216) — +48% Dutch retrieval vs MiniLM
# Output: data/embeddings/bge_m3_embeddings.npy + bge_m3_metadata.json
#
# Run this ONCE before any model experiments to cache all chunk embeddings.
# All subsequent experiments load from cache (no re-embedding needed).

echo "============================="
echo "Job: BGE-M3 Embedding Pre-computation"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================="

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/thesis_env/bin/activate

PROJECT_ROOT="$HOME/Downloads/THESIS"
cd "$PROJECT_ROOT"
mkdir -p logs data/embeddings

python - <<'EOF'
"""Pre-compute and cache BGE-M3 embeddings for all transcript chunks."""
import json
import sys
from pathlib import Path

_ROOT = Path(".")
sys.path.insert(0, str(_ROOT / "experiments" / "01_pipeline"))

from data_loader import merge_csv_sources, prepare_all_samples
from rag_pipeline import speaker_aware_chunk, _get_bge_encoder

data_sources = [
    "data/real/supabase_responses.csv",
    "data/synthetic/synthetic_interviews.csv",
    "data/synthetic/supbase_english_synthetic.csv",
    "data/synthetic/supbase_dutch_synthetic.csv",
]
existing = [p for p in data_sources if Path(p).exists()]
print(f"Loading from {len(existing)} data sources...")

interviews = merge_csv_sources(existing)
print(f"Total interviews: {len(interviews)}")

encoder = _get_bge_encoder(device="cuda")
all_metadata = []

import numpy as np
all_embeddings = []

for iv in interviews:
    transcript_text = " ".join(
        f"{u['speaker']}: {u['text']}" for u in iv.get("utterances", [])
    )
    chunks = speaker_aware_chunk(transcript_text)
    for i, chunk in enumerate(chunks):
        all_metadata.append({
            "interview_id": iv["id"],
            "chunk_idx": i,
            "chunk_text": chunk[:200],  # first 200 chars for metadata
        })

    embeddings = encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    all_embeddings.extend(embeddings.tolist())

print(f"Total chunks embedded: {len(all_embeddings)}")

out_dir = Path("data/embeddings")
np.save(str(out_dir / "bge_m3_embeddings.npy"), np.array(all_embeddings))
with open(out_dir / "bge_m3_metadata.json", "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"Saved embeddings → data/embeddings/bge_m3_embeddings.npy")
print(f"Saved metadata   → data/embeddings/bge_m3_metadata.json")
EOF

echo "============================="
echo "BGE-M3 embedding job COMPLETE"
echo "============================="
