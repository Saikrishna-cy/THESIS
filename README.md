# Hallucination Detection in RAG-Based Systems for Bilingual Interview Transcripts

**Leiden University · Master's Thesis · 2025-2026 · Saikrishna Cynisetty**

---

## What This Project Does

This pipeline automatically generates answers to research questions by retrieving relevant passages from Dutch and English interview transcripts (RAG — Retrieval-Augmented Generation), then detects and classifies hallucinations in those answers using a 6-type taxonomy. Nine language models are benchmarked, including a novel **Dutch training-stage chain** that reveals at which step of the fine-tuning process Dutch hallucinations emerge. Advanced modules apply iterative correction and QLoRA fine-tuning to reduce hallucination rates.

---

## Model Lineup — 9 Models

| Key | Model | Size | GPU | Notes |
|---|---|---|---|---|
| `gpt-4o-mini` | GPT-4o-mini | ~8B eq. | None (API) | Commercial baseline |
| `mistral_base` | Mistral-7B-v0.1 *(base)* | 7B | A100-80GB | Dutch chain step 1 — raw base, **no fine-tuning** |
| `mistral` | Mistral-7B-Instruct-v0.3 | 7B | A100-80GB | Instruction-tuned Mistral |
| `qwen` | Qwen2.5-7B-Instruct | 7B | A100-80GB | Scale baseline |
| `qwen14b` | Qwen2.5-14B-Instruct | 14B | A100-80GB | Scale ablation (4-bit quantized) |
| `geitje_sft` | GEITje-7B-ultra-sft *(SFT)* | 7B | A100-80GB | Dutch chain step 2 — SFT only, before DPO |
| `geitje` | GEITje-7B-ultra *(DPO)* | 7B | A100-80GB | Dutch chain step 3 — final model |
| `aya23` | Aya-23-8B | 8B | A100-80GB | Multilingual (23 languages incl. Dutch) |
| `mixtral` | Mixtral-8×7B-Instruct | 46.7B/12.9B active | A100-80GB | Mixture-of-Experts (4-bit quantized) |

> **Dutch Training-Stage Chain** (`mistral_base` → `geitje_sft` → `geitje`):
> Traces hallucinations across training stages to answer: *"Does Dutch hallucination originate in pretraining, SFT, or DPO alignment?"*
> Recommended by **Bram Vanroy** (GEITje author, personal communication March 2026).
> Note: `mistral_base` has no instruction tuning — raw/unformatted outputs are expected. This is intentional.

---

## Hallucination Types — 6-Type Taxonomy

| # | Type | Description |
|---|---|---|
| 1 | `EVIDENT_CONFLICT` | Model directly contradicts a statement in the transcript |
| 2 | `SUBTLE_CONFLICT` | Plausible but slightly different from what was actually said |
| 3 | `BASELESS_INFO` | Model adds details that do not appear anywhere in the transcript |
| 4 | `SENTIMENT_MISREPRESENTATION` | Wrong sentiment polarity (positive presented as negative, or vice versa) |
| 5 | `REFUSAL_HALLUCINATION` | Model says information is "not available" when it IS present in the transcript |
| 6 | `ROLE_ATTRIBUTION_DRIFT` | Correctly attributes sentence 1 to Speaker A, then drifts to attributing later sentences to Speaker B |

Types 5 and 6 are novel contributions specific to multi-turn bilingual interview RAG.

---

## Pipeline Overview

```
STEP 0  Generate Dataset     600 bilingual interviews (300 EN + 300 NL), 7 RAG queries each
   |
STEP 1  Generate Responses   Each model answers 7 questions per interview via RAG
   |                         (BGE-M3 two-stage retrieval + Evidence CoT prompt)
STEP 2  Detect (RQ1)         GPT-4o-mini judge annotates each response → FAITHFUL or 1-6 types
   |
STEP 3  Benchmark (RQ2)      SelfCheckGPT, NLI detector, RAGAS scored against human labels
   |
STEP 4  Compare              Cross-model hallucination rate table
   |
STEP 5  Unified Results      All model outputs merged into one JSONL file
   |
STEP 6  Statistics           McNemar significance tests + 95% Wilson confidence intervals
   |
[Optional]
  Correction Loop            Iterative self-refinement (max 2 rounds) on hallucinated responses
  QLoRA Fine-Tuning          Fine-tune GEITje on FAITHFUL examples (rank-16 LoRA, NF4 4-bit)
```

---

## Prerequisites

```bash
# Python version
python --version   # requires 3.11+

# Install dependencies
pip install -r requirements.txt

# Create .env file in project root:
OPENAI_API_KEY=sk-...        # required: GPT-4o-mini + hallucination judge
HUGGINGFACE_TOKEN=hf_...     # required: downloading HuggingFace models on ALICE
```

For HuggingFace models (`mistral`, `qwen`, `qwen14b`, `geitje`, `geitje_sft`, `mistral_base`, `aya23`, `mixtral`): **ALICE HPC cluster access** (Leiden University) is required. These models need an A100-80GB GPU.

---

## Quick Start — Local Test (No GPU)

Test the full pipeline locally using GPT-4o-mini on 3 interviews (takes ~5 minutes):

```bash
# Step 0: Generate the controlled dataset
python experiments/00_data_generation/generate_controlled_dataset_v3.py

# Step 1-6: Run the full pipeline on 3 interviews (smoke test)
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini \
    --max-interviews 3 \
    --steps all
```

Results appear in `results/gpt-4o-mini/`.

---

## Full Pipeline — Step by Step

### Step 0 — Generate Controlled Dataset

```bash
python experiments/00_data_generation/generate_controlled_dataset_v3.py
```

Generates `data/controlled/controlled_interviews.json` — 600 bilingual interviews (300 English + 300 Dutch), 14 turns each, 7 RAG query types per interview. Run this once before any pipeline step.

---

### Step 1 — Generate RAG Responses

```bash
# Single model
python experiments/01_pipeline/run_all.py --models gpt-4o-mini --steps generate

# Multiple models (comma-separated)
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini,mistral,qwen \
    --steps generate

# With vLLM backend (2-4x faster on ALICE)
python experiments/01_pipeline/run_all.py \
    --models geitje \
    --steps generate \
    --inference vllm

# Limit interviews for testing
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini \
    --steps generate \
    --max-interviews 10
```

Output: `results/{model}/01_rag_responses.json`

---

### Step 2 — Detect Hallucinations (RQ1)

```bash
# Default judge: gpt-4o-mini (auto-switches if judge == generator to prevent bias)
python experiments/01_pipeline/run_all.py --models gpt-4o-mini --steps rq1

# Custom judge model
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini \
    --steps rq1 \
    --judge-model gpt-4o
```

Output: `results/{model}/02_rq1_annotations.json`

> **Auto-guard**: if the generating model equals the judge model, the pipeline automatically switches to a fallback judge to prevent self-evaluation bias (Zheng et al. 2023, arXiv:2306.05685).

---

### Step 3 — Benchmark Automated Detectors (RQ2)

```bash
python experiments/01_pipeline/run_all.py --models gpt-4o-mini --steps rq2
```

Output: `results/{model}/03_rq2_detectors.json`

---

### Step 4 — Cross-Model Comparison

```bash
python experiments/01_pipeline/run_all.py --steps compare
```

Output: `results/comparison/`

---

### Step 5 — Build Unified Results

```bash
python experiments/01_pipeline/run_all.py --steps unified
```

Output: `data/unified_results.jsonl` — all models in one file, used for statistical analysis.

---

### Step 6 — Statistical Analysis

```bash
python experiments/01_pipeline/run_all.py --steps stats
```

Runs McNemar significance tests and 95% Wilson confidence intervals across all models.

---

### Run All Steps at Once

```bash
# All 9 models, all steps
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini,mistral_base,mistral,qwen,qwen14b,geitje_sft,geitje,aya23,mixtral \
    --steps all
```

---

## ALICE HPC — SLURM Jobs

All HuggingFace models require an A100-80GB GPU. Submit via SLURM:

```bash
# Run once before any HF model job — pre-computes BGE-M3 embeddings
sbatch jobs/precompute_embeddings.sh

# Individual model jobs
sbatch jobs/run_mistral7b.sh      # Mistral-7B-Instruct-v0.3  (12h wall time)
sbatch jobs/run_geitje_sft.sh     # GEITje-SFT  — Dutch chain step 2  (12h)
sbatch jobs/run_geitje.sh         # GEITje-DPO  — Dutch chain step 3  (12h)
sbatch jobs/run_aya23.sh          # Aya-23-8B  (12h)
sbatch jobs/run_qwen14b.sh        # Qwen2.5-14B 4-bit  (16h)
sbatch jobs/run_mixtral.sh        # Mixtral-8×7B 4-bit  (16h)
sbatch jobs/run_qlora.sh          # QLoRA fine-tuning  (8h)

# Monitor your jobs
squeue -u $USER

# Watch job output in real time
tail -f logs/geitje_<JOBID>.out
```

### Dutch Training-Stage Chain (submit all 3 together)

```bash
# Chain: mistral_base → geitje_sft → geitje
# Each job is independent — submit simultaneously or in any order
sbatch jobs/run_mistral7b.sh    # uses --models mistral_base internally
sbatch jobs/run_geitje_sft.sh   # GEITje SFT checkpoint
sbatch jobs/run_geitje.sh       # GEITje DPO final
```

Results can then be compared side-by-side using `--steps compare`.

> **Note**: `mistral_base` uses `mistralai/Mistral-7B-v0.1` with no instruction tuning. Responses will be raw/unformatted — this is expected and intentional for the training-stage comparison.

---

## QLoRA Fine-Tuning

Fine-tunes GEITje on FAITHFUL-labeled examples to reduce its 65.9% hallucination rate.

```bash
# Dry run — validates setup without GPU (run locally first)
python experiments/06_finetuning/run_qlora.py --dry-run

# Full run on ALICE A100-80GB (recommended)
sbatch jobs/run_qlora.sh

# Custom options
python experiments/06_finetuning/run_qlora.py \
    --model geitje \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3
```

Architecture: NF4 4-bit quantization + LoRA rank-16 on Q/K/V/O layers.
Training data: FAITHFUL-labeled responses from `02_rq1_annotations.json`.
Output: `results/finetuned/geitje_qlora/lora_adapter/`

---

## Output Files Reference

```
results/
  {model}/
    01_rag_responses.json       RAG-generated answers with retrieved transcript context
    02_rq1_annotations.json     Hallucination labels per response (FAITHFUL or type 1-6)
    03_rq2_detectors.json       Automated detector benchmark scores
    role_drift_counts.json      ROLE_ATTRIBUTION_DRIFT analysis (heuristic post-processing)

  comparison/                   Cross-model hallucination rate tables and charts
  finetuned/geitje_qlora/       QLoRA adapter weights + training config

data/
  controlled/                   Generated interviews (gitignored — run Step 0 first)
  unified_results.jsonl         All models merged into one file (for statistics)
```

---

## Repository Structure

```
THESIS/
├── experiments/
│   ├── 00_data_generation/    STEP 0  — generate 600 bilingual controlled interviews
│   ├── 01_pipeline/           STEP 1  — RAG generation + 9-model registry
│   ├── 02_rq1/                STEP 2  — hallucination taxonomy annotation (6 types)
│   ├── 03_rq2/                STEP 3  — automated detector benchmarking
│   ├── 04_advanced/           ADVANCED — iterative correction loop (max 2 rounds)
│   └── 06_finetuning/         ADVANCED — QLoRA fine-tuning (rank-16, NF4 4-bit)
│
├── jobs/                      SLURM job scripts for ALICE A100-80GB HPC
│   ├── run_mistral7b.sh       Mistral-7B-Instruct-v0.3
│   ├── run_geitje_sft.sh      GEITje-SFT  (Dutch chain step 2)
│   ├── run_geitje.sh          GEITje-DPO  (Dutch chain step 3)
│   ├── run_aya23.sh           Aya-23-8B
│   ├── run_qwen14b.sh         Qwen2.5-14B (4-bit)
│   ├── run_mixtral.sh         Mixtral-8×7B (4-bit)
│   ├── run_qlora.sh           QLoRA fine-tuning
│   ├── precompute_embeddings.sh  BGE-M3 embedding pre-computation
│   └── run_vllm_base.sh       Generic vLLM template
│
├── data/                      Dataset (generate before running — see Step 0)
│   ├── controlled/            Generated interviews (gitignored)
│   ├── real/                  Placeholder (old CSV data removed)
│   └── synthetic/             Placeholder (old CSV data removed)
│
├── results/                   Pipeline outputs (gitignored)
├── docs/                      Architecture diagrams + PDF how-to guide
├── utils/                     Shared utilities (stats, comparison, unified builder)
├── HOW_TO_RUN.md              Detailed step-by-step guide
├── CHANGELOG.txt              Every code change with research paper backing
└── requirements.txt           Python dependencies
```

---

## CLI Reference

```
python experiments/01_pipeline/run_all.py [OPTIONS]

Options:
  --models       Comma-separated model keys (default: all 9)
  --steps        Pipeline steps: all | generate | rq1 | rq2 | compare | unified | stats
  --max-interviews N   Limit to N interviews (useful for testing)
  --inference    transformers (default) | vllm (2-4x faster on ALICE)
  --judge-model  OpenAI model for hallucination annotation (default: gpt-4o-mini)
  --results-base Output directory base (default: results/)
  --selfcheck-samples N   SelfCheckGPT samples per response (default: 5)
```

---

## Key Papers

1. Chen et al. 2024 (arXiv:2402.03216) — BGE-M3 two-stage retrieval (bi-encoder + reranker)
2. Dettmers et al. 2023 (arXiv:2305.17333) — QLoRA: 4-bit NF4 quantization
3. Hu et al. 2021 (arXiv:2106.09685) — LoRA: low-rank adaptation adapters
4. Jiang et al. 2023 (arXiv:2310.06825) — Mistral-7B: GQA + sliding window attention
5. Jiang et al. 2024 (arXiv:2401.04088) — Mixtral-8×7B: Mixture-of-Experts architecture
6. Kwon et al. 2023 (arXiv:2309.06180) — vLLM: PagedAttention for fast inference
7. Madaan et al. 2023 (arXiv:2303.17651) — Self-Refine: iterative self-correction
8. Ouyang et al. 2022 (arXiv:2203.02155) — InstructGPT: SFT vs RLHF training stages
9. Shi et al. 2023 (arXiv:2108.12409) — Lost in the Middle: positional decay in LLMs
10. Zheng et al. 2023 (arXiv:2306.05685) — LLM-as-judge: self-evaluation bias
