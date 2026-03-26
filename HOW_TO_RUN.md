# How to Run the Hallucination Detection Pipeline

## Prerequisites

- **Python 3.11** (tested with 3.11.3)
- **OpenAI API key** — required for GPT-4o-mini generation and all annotation steps
- **ALICE HPC account** — required for running open-source models (Mistral, Qwen, GEITje, Aya-23, Mixtral)
- **HuggingFace token** — optional; only needed if you add gated models (LLaMA)

### Python dependencies

```bash
# Create virtual environment
python3.11 -m venv ~/thesis_env
source ~/thesis_env/bin/activate

# Core dependencies
pip install openai python-dotenv sentence-transformers transformers accelerate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes  # for 4-bit quantization (qwen14b, mixtral)

# Optional: vLLM for 2-4x speedup
pip install vllm

# Optional: QLoRA fine-tuning
pip install peft trl datasets
```

### Environment setup

Create a `.env` file in the project root (`~/Downloads/THESIS/.env`):

```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...  # optional
```

---

## Quick Local Test (GPT-4o-mini only, 3 interviews)

This requires only an OpenAI API key — no GPU needed.

```bash
cd ~/Downloads/THESIS

# Run full pipeline on 3 interviews with GPT-4o-mini
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini \
    --steps generate,rq1,rq2 \
    --max-interviews 3

# Check results
ls results/gpt-4o-mini/
# 01_rag_responses.json
# 02_rq1_annotations.json
# 03_rq1_huang_taxonomy.json
# 04_rq1_diahalu_eval.json
```

### Run drift detection (post-processing, no API calls)

```bash
python experiments/02_rq1/detect_role_drift.py --model gpt-4o-mini
```

### Run correction loop

```bash
python experiments/04_advanced/correction_loop.py
# Results saved to results/correction_loop/correction_results.json
```

---

## Full ALICE Run (All 7 Models)

### Step 1: Pre-compute embeddings (run once)

```bash
sbatch jobs/precompute_embeddings.sh
# Wait for completion (~30-60 min)
squeue -u $USER
```

### Step 2: Run all models in parallel

```bash
# API model (no GPU, run locally or as a CPU job)
python experiments/01_pipeline/run_all.py \
    --models gpt-4o-mini \
    --steps generate,rq1,rq2

# Open models — submit as separate SLURM jobs
sbatch jobs/run_mistral7b.sh
sbatch jobs/run_qwen14b.sh
sbatch jobs/run_mixtral.sh

# Generic template for other models
sbatch --export=MODEL=geitje  jobs/run_vllm_base.sh
sbatch --export=MODEL=aya23   jobs/run_vllm_base.sh
sbatch --export=MODEL=qwen    jobs/run_vllm_base.sh
```

### Step 3: Monitor jobs

```bash
squeue -u $USER
# Check logs
tail -f logs/mistral7b_<JOBID>.out
```

### Step 4: Cross-model comparison and statistics

```bash
python experiments/01_pipeline/run_all.py \
    --steps unified,compare,stats \
    --results-base results
```

---

## Individual Model SLURM Jobs

| Model | Script | Wall Time | VRAM |
|---|---|---|---|
| Mistral-7B | `jobs/run_mistral7b.sh` | 12h | ~14GB |
| Qwen2.5-7B | `jobs/run_vllm_base.sh MODEL=qwen` | 4h (vLLM) | ~14GB |
| Qwen2.5-14B | `jobs/run_qwen14b.sh` | 16h | ~11GB (4-bit) |
| GEITje-7B | `jobs/run_vllm_base.sh MODEL=geitje` | 4h (vLLM) | ~14GB |
| Aya-23-8B | `jobs/run_vllm_base.sh MODEL=aya23` | 4h (vLLM) | ~16GB |
| Mixtral-8x7B | `jobs/run_mixtral.sh` | 16h | ~24GB (4-bit) |

---

## Step-by-Step Pipeline Steps

The pipeline has 6 steps, controlled by `--steps`:

| Step | What it does | Output files |
|---|---|---|
| `generate` | RAG responses for all interviews | `01_rag_responses.json` |
| `rq1` | Hallucination taxonomy annotation (GPT-4o-mini judge) | `02_rq1_annotations.json`, `03_rq1_huang_taxonomy.json`, `04_rq1_diahalu_eval.json` |
| `rq2` | Hallucination detection benchmarks | `05_rq2_detection.json` |
| `compare` | Cross-model comparison tables | `results/comparison/` |
| `unified` | Merge all models into single JSONL | `data/unified_results.jsonl` |
| `stats` | McNemar tests + 95% CIs | `statistics_report.json` |

Run specific steps:

```bash
# Only annotation (responses already exist)
python experiments/01_pipeline/run_all.py --models qwen --steps rq1

# Only statistics (all results already exist)
python experiments/01_pipeline/run_all.py --steps unified,stats
```

---

## Running QLoRA Fine-tuning

Fine-tuning requires the RQ1 annotation step to be complete first.

### Dry run (local, no GPU)

```bash
python experiments/06_finetuning/run_qlora.py \
    --model geitje \
    --dry-run
```

### Full fine-tuning on ALICE

```bash
# Default: GEITje-7B with rank-16 LoRA, 3 epochs
sbatch jobs/run_qlora.sh

# Custom: Aya-23 with rank-32
sbatch --export=MODEL=aya23,LORA_R=32,LORA_ALPHA=64,EPOCHS=5 jobs/run_qlora.sh
```

Output is saved to `results/finetuned/{model}_qlora/`:
- `lora_adapter/` — LoRA weights (load with `peft.PeftModel.from_pretrained`)
- `training_config.json` — all hyperparameters

---

## Output Files Explanation

After running the full pipeline for one model (e.g., `gpt-4o-mini`):

```
results/
  gpt-4o-mini/
    01_rag_responses.json          # RAG-generated responses + retrieved context
    02_rq1_annotations.json        # Hallucination type labels per response
    03_rq1_huang_taxonomy.json     # Huang et al. taxonomy classification
    04_rq1_diahalu_eval.json       # Dialogue-level hallucination evaluation
    05_rq2_detection.json          # RQ2 automated detection scores
    role_drift_counts.json         # ROLE_ATTRIBUTION_DRIFT heuristic detection
  correction_loop/
    correction_results.json        # Before/after correction statistics
  comparison/
    cross_model_comparison.json    # Side-by-side model comparison
  finetuned/
    geitje_qlora/
      lora_adapter/                # Fine-tuned LoRA weights
      training_config.json         # Training hyperparameters
```

### Key metrics in `02_rq1_annotations.json`

Each entry has:
- `overall_label`: `"HALLUCINATED"` or `"FAITHFUL"`
- `faithfulness_score`: 0.0–1.0
- `hallucinations`: list of `{type, span, evidence, severity}`
- `hallucination_types`: list of type names (for aggregation)

### Interpreting hallucination types

| Type | Description |
|---|---|
| `EVIDENT_CONFLICT` | Direct contradiction of transcript |
| `SUBTLE_CONFLICT` | Minor deviation from transcript |
| `BASELESS_INFO` | Claim with no transcript grounding |
| `SENTIMENT_MISREPRESENTATION` | Wrong emotional tone attributed |
| `REFUSAL_HALLUCINATION` | "Not available" when answer IS in transcript |
| `ROLE_ATTRIBUTION_DRIFT` | Progressive drift from correct to wrong speaker |
