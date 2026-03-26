# Thesis: Hallucination Detection in Interview-Based RAG Systems

## Overview

This repository implements a **6-model, multilingual, multi-method** hallucination detection study for RAG-based interview analysis systems. The study spans two research questions:

- **RQ1:** What types of hallucinations occur in RAG-generated interview analyses?
- **RQ2:** Which automated detection methods best identify these hallucinations?

The thesis makes five novel contributions:
1. First hallucination taxonomy study in the interview RAG domain
2. Multi-taxonomy comparison (RAGTruth + Huang + DiaHaLu) on real interview data
3. Discovery of REFUSAL_HALLUCINATION as a systematic GPT-4o-mini behaviour
4. First Dutch-specific RAG hallucination study (GEITje-7B-Ultra vs multilingual Aya-23-8B)
5. Learned ensemble detector beating all individual methods

---

## Model Lineup

| Key | Model | Type | GPU | Dutch? |
|-----|-------|------|-----|--------|
| `gpt-4o-mini` | GPT-4o-mini | OpenAI API | No | — |
| `mistral` | Mistral-Small-24B | Mistral API | No | — |
| `qwen` | Qwen2.5-7B-Instruct | HuggingFace | A100 40GB | — |
| `llama` | Llama-3.1-8B-Instruct | HuggingFace | A100 40GB | — |
| `geitje` | GEITje-7B-Ultra | HuggingFace | A100 40GB | ★ Dutch |
| `aya23` | Aya-23-8B | HuggingFace | A100 40GB | ✓ multilingual |

---

## Quick Start

### 1. Setup

```bash
git clone <repo>
cd THESIS-main

# Create virtualenv
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy .env template and add your keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY, MISTRAL_API_KEY, HF_TOKEN (for Aya-23)
```

### 2. Run API models (no GPU needed)

```bash
# Full pipeline for GPT-4o-mini + Mistral:
python experiments/01_pipeline/run_all.py \
  --models gpt-4o-mini,mistral \
  --steps all

# Test with just 5 interviews first:
python experiments/01_pipeline/run_all.py \
  --models gpt-4o-mini \
  --steps generate \
  --max-interviews 5
```

### 3. Run on ALICE HPC (HuggingFace models)

```bash
# Dutch model (GEITje):
sbatch jobs/run_geitje.sh

# Multilingual model (Aya-23):
sbatch jobs/run_aya23.sh

# All HF models in one job:
sbatch jobs/generate_responses.sh

# Monitor:
squeue -u $USER
tail -f logs/geitje_<JOBID>.out
```

### 4. Run full analysis pipeline

```bash
# After responses are generated for all models:
python experiments/01_pipeline/run_all.py --steps rq1,rq2,compare,unified,stats

# Build unified JSONL (needed for advanced experiments):
python utils/build_unified_results.py

# Statistical analysis:
python utils/statistics_utils.py --input data/unified_results.jsonl
```

### 5. Advanced experiments (run after unified JSONL exists)

```bash
# Ensemble detector (requires manual_validation_200.csv to be labelled):
python experiments/04_advanced/ensemble_detector.py

# Chunk overlap × MMR ablation (key novel experiment):
python experiments/06_hyperparameter/chunk_overlap_analysis.py \
  --model gpt-4o-mini --max-interviews 30

# Manual validation kappa:
python experiments/07_validation/calculate_kappa.py
```

### 6. Generate PDF reports

```bash
python reports/generate_thesis_report.py
python reports/generate_companion_guide.py
# → results/thesis_ieee_report.pdf
# → results/thesis_companion_guide.pdf
```

---

## Expected Run Order

```
data/ (CSVs)
    ↓
experiments/01_pipeline/run_all.py --steps generate    # ~8-12h per HF model on A100
    ↓
experiments/01_pipeline/run_all.py --steps rq1         # ~2-4h per model (GPT-4o judge)
    ↓
experiments/01_pipeline/run_all.py --steps rq2         # ~4-8h per model (4 detectors)
    ↓
utils/build_unified_results.py                         # <1 min
    ↓
utils/statistics_utils.py                              # <1 min
    ↓
experiments/04_advanced/ensemble_detector.py            # <5 min
experiments/06_hyperparameter/chunk_overlap_analysis.py # ~2h per model
experiments/07_validation/calculate_kappa.py            # <1 min
    ↓
reports/generate_thesis_report.py                      # <2 min
reports/generate_companion_guide.py                    # <2 min
```

---

## Data Files

| File | Description | Rows |
|------|-------------|------|
| `data/real/supabase_responses.csv` | Real interview transcripts from Supabase | ~200 |
| `data/synthetic/synthetic_interviews.csv` | English synthetic interviews | ~150 |
| `data/synthetic/supbase_english_synthetic.csv` | Additional English synthetic | ~100 |
| `data/synthetic/supbase_dutch_synthetic.csv` | Dutch synthetic interviews | ~150 |
| `data/unified_results.jsonl` | Generated: all model results merged | ~9,000+ |
| `RAG_THESIS/output/manual_validation_200.csv` | 198 stratified samples for annotation | 198 |

---

## Key Results (from prior runs — to be regenerated)

| Model | Hallucination Rate | Key Finding |
|-------|-------------------|-------------|
| GPT-4o-mini | 20.77% | 60% are REFUSAL_HALLUCINATION |
| Mistral-Small-24B | 17.54% | Best API model (McNemar p=0.0001 vs GPT) |
| Qwen2.5-7B | 22.15% | Highest rate; strongest for Dutch queries |
| GEITje-7B-Ultra | TBD (new) | Expected lower on Dutch interviews |
| Aya-23-8B | TBD (new) | Multilingual Dutch baseline |
| Llama-3.1-8B | TBD (new) | Reproducibility baseline |

| Detector | Best F1 (GPT-4o-mini) | Note |
|----------|----------------------|------|
| SelfCheckGPT | 0.000 | Fails in dialogue domain |
| MiniCheck | 0.857 | Best individual detector |
| AlignScore | 0.432 | Moderate |
| RAGAS | 0.389 | Moderate |
| **Ensemble** | **0.871** | Beats all individuals |

---

## Troubleshooting

**ImportError for HuggingFace models:**
```bash
pip install transformers accelerate sentencepiece protobuf
```

**CUDA out of memory on A100:**
```bash
# Add to rag_pipeline.py _generate_hf(): load_in_4bit=True via bitsandbytes
pip install bitsandbytes
```

**Aya-23 access denied (gated model):**
```bash
# Accept terms at: https://huggingface.co/CohereForAI/aya-23-8B
huggingface-cli login --token YOUR_HF_TOKEN
```

**Missing .env keys:**
```bash
# Required for API models + embeddings:
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
# Optional:
TOGETHER_API_KEY=...   # fallback for Qwen via Together.ai
HF_TOKEN=hf_...        # for gated HuggingFace models
```

---

## File Structure

See `docs/PIPELINE_CONNECTIONS.md` for the complete file dependency map.

```
THESIS-main/
├── experiments/
│   ├── 01_pipeline/     ← data loading, RAG generation, orchestration
│   ├── 02_rq1/          ← hallucination taxonomy classification
│   ├── 03_rq2/          ← automated detector evaluation
│   ├── 04_advanced/     ← ensemble, predictor, correction loop
│   ├── 05_ablation/     ← no-RAG baseline, CoT ablation
│   ├── 06_hyperparameter/ ← sweep: chunk_size, overlap, MMR, prompts
│   └── 07_validation/   ← export CSV, Cohen's Kappa, detector eval
├── utils/               ← shared: embeddings, configs, stats, reports
├── reports/             ← PDF generators (thesis report + companion)
├── jobs/                ← ALICE SLURM job scripts
├── data/                ← interview CSVs + unified_results.jsonl
├── results/             ← generated: one subfolder per model
└── docs/                ← this file + PIPELINE_CONNECTIONS.md
```
