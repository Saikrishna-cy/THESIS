# Pipeline Architecture

## Overview

This pipeline detects, classifies, and mitigates hallucinations in RAG-based qualitative interview analysis systems. It supports 7 LLMs (1 API, 6 open-source) across Dutch and English interview data.

---

## Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│                                                                 │
│  data/real/supabase_responses.csv    (real interview data)      │
│  data/synthetic/synthetic_interviews.csv  (synthetic Dutch)     │
│  data/synthetic/supbase_english_synthetic.csv  (synthetic EN)   │
│  data/synthetic/supbase_dutch_synthetic.csv    (synthetic NL)   │
│                          │                                      │
│              data_loader.merge_csv_sources()                    │
│              data_loader.prepare_all_samples()                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 1: RAG GENERATION                            │
│            experiments/01_pipeline/rag_pipeline.py              │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │  RETRIEVAL      │    │  GENERATION                      │   │
│  │                 │    │                                  │   │
│  │ BGE-M3 encoder  │    │  gpt-4o-mini  → OpenAI API       │   │
│  │ BAAI/bge-m3     │    │  mistral      → HuggingFace HF   │   │
│  │ (dense+sparse)  │    │  qwen         → HuggingFace HF   │   │
│  │       ↓         │    │  qwen14b      → HF + 4-bit NF4   │   │
│  │ top-20 chunks   │    │  geitje       → HuggingFace HF   │   │
│  │       ↓         │    │  aya23        → HuggingFace HF   │   │
│  │ bge-reranker    │    │  mixtral      → HF + 4-bit NF4   │   │
│  │ cross-encoder   │    │                                  │   │
│  │       ↓         │    │  Backend: Transformers or vLLM   │   │
│  │ top-3 context   │    │  System prompt: evidence_cot     │   │
│  └────────┬────────┘    └──────────────┬───────────────────┘   │
│           └──────────────┬─────────────┘                       │
│                          ▼                                      │
│              01_rag_responses.json (per model)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 2: RQ1 HALLUCINATION TAXONOMY (annotation)        │
│            experiments/02_rq1/experiment_rq1.py                 │
│                                                                 │
│  EVALUATOR: GPT-4o-mini (judges ALL model outputs)              │
│                                                                 │
│  Experiment 1 — RAGTruth-style annotation                       │
│    → 6 hallucination types (novel taxonomy):                    │
│      1. EVIDENT_CONFLICT                                        │
│      2. SUBTLE_CONFLICT                                         │
│      3. BASELESS_INFO                                           │
│      4. SENTIMENT_MISREPRESENTATION                             │
│      5. REFUSAL_HALLUCINATION                                   │
│      6. ROLE_ATTRIBUTION_DRIFT  ← novel contribution           │
│    Output: 02_rq1_annotations.json                              │
│                                                                 │
│  Experiment 2 — Huang et al. taxonomy mapping                   │
│    → FACTUALITY vs FAITHFULNESS + interview-specific types      │
│    Output: 03_rq1_huang_taxonomy.json                           │
│                                                                 │
│  Experiment 3 — DiaHaLu dialogue evaluation                     │
│    → 6 dialogue-specific issues (cross-turn, entity confusion)  │
│    Output: 04_rq1_diahalu_eval.json                             │
│                                                                 │
│  Post-processing (no API calls):                                │
│    detect_role_drift.py → role_drift_counts.json                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 3: RQ2 HALLUCINATION DETECTION                    │
│            experiments/03_rq2/experiment_rq2.py                 │
│                                                                 │
│  Compares automated detection methods:                          │
│  - SelfCheckGPT (Manakul et al. 2023, arXiv:2303.08896)        │
│  - NLI-based detection (Honovich et al. 2022)                   │
│  - Semantic similarity (BGE-M3 cosine)                          │
│  Output: 05_rq2_detection.json                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 4: ADVANCED — CORRECTION LOOP                     │
│            experiments/04_advanced/correction_loop.py           │
│                                                                 │
│  For each HALLUCINATED response:                                │
│    Round 1: Error-aware prompt → corrected response             │
│             → re-annotate → if FAITHFUL, stop                   │
│    Round 2: (if still HALLUCINATED) repeat with new errors      │
│                                                                 │
│  Theory: Madaan et al. 2023 Self-Refine (arXiv:2303.17651)      │
│  Output: results/correction_loop/correction_results.json        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 5: STATISTICAL ANALYSIS                           │
│            utils/statistics_utils.py                            │
│                                                                 │
│  - McNemar test: pairwise model significance                    │
│  - Wilson confidence intervals (95%) per model                  │
│  - Scale ablation: qwen7B vs qwen14B (McNemar)                  │
│  Output: statistics_report.json                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 6: QLORA FINE-TUNING (optional)                   │
│            experiments/06_finetuning/run_qlora.py               │
│                                                                 │
│  Input: FAITHFUL-labeled examples from 02_rq1_annotations.json  │
│  Method: QLoRA (Dettmers et al. 2023, arXiv:2305.17333)         │
│    - NF4 4-bit quantization                                     │
│    - LoRA rank-16 on Q/K/V/O layers                             │
│    - SFT on faithful responses                                  │
│  Goal: reduce hallucination rate below 5%                       │
│  Output: results/finetuned/{model}_qlora/lora_adapter/          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### Data Layer

- `data_loader.py`: Merges CSV sources, normalises column names, builds per-interview sample dicts with `interview_id`, `language`, `utterances`, and query variants.
- Data sources: 1 real (Supabase export) + 3 synthetic (Dutch + English).

### RAG Pipeline (`rag_pipeline.py`)

**Chunking**: `speaker_aware_chunk()` splits at speaker turn boundaries (regex: `(?=(?:Interviewer|Participant|Agent|Speaker\d*):)`), targeting 400-token chunks with 50-token overlap. Preserves full speaker turns — critical for attribution correctness.

**Retrieval**: Two-stage BGE-M3 pipeline:
1. Bi-encoder (BAAI/bge-m3): encodes all chunks, retrieves top-20 by cosine similarity
2. Cross-encoder (BAAI/bge-reranker-v2-m3): reranks top-20, returns top-3

**Generation**: Evidence CoT system prompt forces 3-step grounding:
1. QUOTE relevant transcript sections
2. VERIFY quote supports the query
3. ANSWER based only on verified quotes

**Backends**: Transformers (default) or vLLM (2-4x faster via PagedAttention).

### Model Registry (`model_registry.py`)

Stores HuggingFace IDs, VRAM estimates, quantization config, and ALICE SLURM parameters for all 7 models.

### Hallucination Detection (`experiment_rq1.py`)

Uses GPT-4o-mini as a judge to annotate outputs from any model. The novel taxonomy adds ROLE_ATTRIBUTION_DRIFT to capture a pattern not in existing benchmarks.

### Role Drift Detector (`detect_role_drift.py`)

Heuristic post-processing — zero API calls. Uses regex speaker labelling + positional analysis. Grounded in "Lost in the Middle" positional decay theory (Shi et al. 2023, arXiv:2108.12409).

### Correction Loop (`correction_loop.py`)

Iterative self-refinement (max 2 rounds). For each HALLUCINATED response, generates an error-aware correction prompt, gets a new response, re-annotates. Early exits when FAITHFUL is achieved.

### QLoRA Fine-tuning (`run_qlora.py`)

Supervised fine-tuning on FAITHFUL-labeled outputs. Uses NF4 4-bit base model + LoRA rank-16 adapters on attention layers only. Trains for 3 epochs with cosine LR schedule.

---

## Where RAG Is Used

RAG is used in **Step 1 only** (generation). The retrieval + generation loop:
- Retrieves top-3 context chunks from the interview transcript
- Feeds context + query to the LLM
- LLM generates a grounded response

All subsequent steps (annotation, detection, correction) use the saved `rag_response` text from `01_rag_responses.json`.

---

## Where Hallucination Detection Happens

Detection happens in two layers:

1. **LLM-as-judge** (Steps 2-3): GPT-4o-mini compares response against transcript and assigns hallucination types + severity.
2. **Heuristic post-processing** (detect_role_drift.py): regex-based speaker labelling — no LLM needed.

---

## Where QLoRA Fine-tuning Fits

Fine-tuning is an **optional post-pipeline step** (Step 6). It requires:
- Completed Steps 1-2 (responses + annotations)
- FAITHFUL-labeled examples as training data
- A100-80GB GPU (ALICE HPC) for 8h

The fine-tuned adapter is evaluated by re-running Steps 1-2 with the fine-tuned model and comparing hallucination rates before vs after.
