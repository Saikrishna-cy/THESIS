# Hallucination in Dutch/English RAG Systems — Full Experiment Architecture

**Leiden University · Master's Thesis · 2025–2026**
**Author:** Sai Krishna Reddy Mulakkayala (s4238206)
**Cluster:** ALICE HPC — Leiden University

> **Research Goal:** Measure, understand, and reduce hallucination in Retrieval-Augmented Generation (RAG)
> systems applied to Dutch and English interview transcripts, across 5 progressive experiment phases.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Data Loading & Dataset Construction](#2-data-loading--dataset-construction)
3. [Experiment 1 — Baseline RAG](#3-experiment-1--baseline-rag)
4. [Experiment 2 — Model Scale (Qwen2.5-14B)](#4-experiment-2--model-scale)
5. [Experiment 3 — DPO Fine-tuning](#5-experiment-3--dpo-fine-tuning)
6. [Experiment 4 — Context-Aware Decoding (CAD)](#6-experiment-4--context-aware-decoding)
7. [Experiment 5 — Three Faithfulness Approaches](#7-experiment-5--three-faithfulness-approaches)
8. [Hallucination Detection & RQ2](#8-hallucination-detection--rq2)
9. [Ablation & Hyperparameter Studies](#9-ablation--hyperparameter-studies)
10. [Validation & Inter-Annotator Agreement](#10-validation--inter-annotator-agreement)
11. [Key Findings & Interpretation](#11-key-findings--interpretation)
12. [Reproducibility & Job Scripts](#12-reproducibility--job-scripts)

---

## 1. Project Structure

```
THESIS-rag_hallucination/
│
├── data/
│   ├── controlled/           # 600 interviews × 6 query types = 3,600 evaluation items
│   ├── real/                 # Real-world interview transcripts
│   └── synthetic/            # GPT-4o-generated synthetic transcripts
│
├── experiments/
│   ├── 00_data_generation/   # generate_controlled_dataset_v3.py
│   ├── 01_pipeline/          # Core RAG: retriever, generator, run_all.py
│   ├── 02_rq1/               # RQ1: annotation & hallucination taxonomy
│   ├── 03_rq2/               # RQ2: automatic detector evaluation
│   ├── 04_advanced/          # Attribution, ensemble, correction loop
│   ├── 05_ablation/          # CoT ablation, no-RAG baseline
│   ├── 06_finetuning/        # DPO / QLora fine-tuning (Experiment 3)
│   ├── 06_hyperparameter/    # Grid search: top-k, chunk-size, temperature
│   ├── 07_validation/        # Human kappa, retrieval grounding, detector eval
│   ├── fourth_results/       # Experiment 4: CAD + evidence-citation prompts
│   ├── fifth_results/        # Experiment 5: extractive, NLI-filter, GPT-4o+NLI
│   ├── local_judge.py        # FREE local Qwen2.5-32B→14B→7B annotation judge
│   └── generate_thesis_report.py
│
├── jobs/                     # SLURM batch scripts for all experiments
├── results/                  # Output JSON per model/approach (gitignored)
└── logs/                     # SLURM stdout/stderr logs (gitignored)
```

---

## 2. Data Loading & Dataset Construction

**Scripts:**
- `experiments/00_data_generation/generate_controlled_dataset_v3.py`
- `experiments/01_pipeline/data_loader.py`

### Dataset Design

The evaluation dataset is **controlled, bilingual, and stratified** — designed to stress-test each hallucination type independently across two languages.

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Interviews     | 600 unique transcripts                     |
| Languages      | Dutch (NL) + English (EN) — ~50% each     |
| Query types    | 6 types × 600 = **3,600 evaluation items** |
| Domain         | University counselling / student life      |
| Avg. transcript length | 14 dialogue turns                  |

### The 6 Query Types

Each interview receives exactly one question per type, chosen to activate distinct failure modes:

| # | Query Type            | What it Requires                          | Primary Hallucination Risk          |
|---|----------------------|------------------------------------------|-------------------------------------|
| 1 | `factual_summary`     | Synthesise multiple facts                | SUBTLE_CONFLICT, BASELESS_INFO      |
| 2 | `participant_content` | What did the participant discuss?        | Paraphrase drift                    |
| 3 | `sentiment`           | What was the speaker's sentiment?        | SENTIMENT_MISREPRESENTATION         |
| 4 | `speaker_attribution` | Who said what?                           | SPEAKER_MISATTRIBUTION              |
| 5 | `specific_content`    | Verbatim retrieval of a specific moment  | TEMPORAL_CONFUSION, BASELESS_INFO   |
| 6 | `temporal`            | What happened before/after event Y?      | TEMPORAL_CONFUSION                  |

### RAG Pipeline Architecture

**Script:** `experiments/01_pipeline/rag_pipeline.py`

```
Interview Transcript (14 turns)
          │
          ▼
    [Text Chunking]
    chunk_size = 512 tokens
    overlap    = 50 tokens
          │
          ▼
    [Dense Embedding]
    Model: BAAI/bge-m3  (multilingual)
          │
          ▼
    [FAISS Vector Index]  ◄─── stored per interview
          │
    Query ──► [BGE-M3 Query Encoder]
          │
          ▼
    [Top-K Retrieval]   k = 5  (tuned via grid search)
          │
          ▼
    [BGE Reranker v2-M3]  (cross-encoder reranking)
          │
          ▼
    [LLM Generation]   (model-specific prompt)
          │
          ▼
    [Response]  ──►  [GPT-4o Judge v2]  ──►  Annotation JSON
```

### GPT-4o Judge (v2) — The Ground Truth Labeller

**Function:** `annotate_rq1()` in `experiments/fourth_results/run_fourth_results.py`

The judge reads the full transcript (up to 20,000 chars / ~5K tokens) alongside the model response and labels each response across **7 hallucination types** with severity:

| Type                        | Definition                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| `EVIDENT_CONFLICT`          | Response directly contradicts a specific statement in the transcript       |
| `SUBTLE_CONFLICT`           | Meaning-changing deviation (e.g. "tried once" → "went regularly")          |
| `BASELESS_INFO`             | Fabricated claim entirely absent from the transcript                       |
| `SPEAKER_MISATTRIBUTION`    | Statement credited to the wrong speaker                                    |
| `TEMPORAL_CONFUSION`        | Events described in wrong chronological order                              |
| `SENTIMENT_MISREPRESENTATION` | Wrong sentiment attributed (enthusiasm → dissatisfaction)               |
| `REFUSAL_HALLUCINATION`     | Claims info unavailable when a specific answer clearly exists              |

**Severity:** LOW / MEDIUM / HIGH per instance.
**Label rule:** `HALLUCINATED` if ≥ 1 MEDIUM or HIGH, or multiple LOW together substantially misrepresent the transcript. Responses with only minor LOW-severity paraphrases that preserve factual meaning are `FAITHFUL`.

**Output JSON format per item:**
```json
{
  "overall_label": "HALLUCINATED",
  "faithfulness_score": 0.3,
  "query_type": "specific_content",
  "hallucinations": [
    {
      "type": "SUBTLE_CONFLICT",
      "span": "the participant visited the counsellor weekly",
      "evidence": "transcript says 'I went once to check it out'",
      "severity": "HIGH",
      "explanation": "Frequency grossly exaggerated"
    }
  ]
}
```

---

## 3. Experiment 1 — Baseline RAG

**Scripts:** `experiments/01_pipeline/run_all.py`, `experiments/02_rq1/experiment_rq1.py`
**Results:** `results/{model}/`

### What This Establishes

This is the **zero-intervention starting point**. Each model receives retrieved chunks as context and generates an answer with a standard RAG prompt. No faithfulness mitigation of any kind.

**Prompt (all models, Experiment 1):**
```
System:
  You are a research assistant answering questions about interview transcripts.
  Use only information from the provided context. Do not add information
  that is not in the transcript.

Context:
  {retrieved_chunks}

Question:
  {query}
```

### Models Evaluated

| Model                          | Size              | Language Focus        |
|-------------------------------|-------------------|-----------------------|
| GPT-4o-mini                   | ~8B (est.)        | EN / multilingual     |
| Mistral-7B-Instruct-v0.3      | 7B                | EN / FR               |
| Qwen2.5-7B-Instruct           | 7B                | EN / ZH / multilingual|
| Aya-23-8B (CohereForAI)       | 8B                | 23 languages incl. NL |
| GEITje-7B-ultra               | 7B                | Dutch-specialised     |
| Mixtral-8x7B-Instruct-v0.1    | 46.7B / 12.9B active | EN / multilingual |

### Results — Overall Hallucination Rate

| Model         | Hallucinated | Valid | Rate      |
|---------------|-------------|-------|-----------|
| GPT-4o-mini   | 1,113       | 3,599 | **30.9%** |
| Mistral-7B    | 1,192       | 3,600 | **33.1%** |
| Qwen2.5-7B    | 1,504       | 3,600 | **41.8%** |
| Aya-23-8B     | 2,147       | 3,584 | **59.9%** |
| GEITje-7B     | 2,508       | 3,600 | **69.7%** |
| Mixtral-8x7B  | —           | —     | Pending (MiniCheck eval) |

### Results — Hallucination Rate by Query Type

| Query Type            | Qwen 7B | Mistral 7B | Aya23  | GEITje |
|-----------------------|---------|-----------|--------|--------|
| `factual_summary`     | 33%     | 66%       | 73%    | 90%    |
| `participant_content` | 7%      | 11%       | 16%    | 46%    |
| `sentiment`           | 37%     | 66%       | 90%    | 96%    |
| `speaker_attribution` | **78%** | **3%**    | 39%    | 51%    |
| `specific_content`    | **83%** | 50%       | **82%**| **99%**|
| `temporal`            | 13%     | 2%        | 60%    | 36%    |

### Dominant Hallucination Types (Top 3 per Model)

| Model      | #1                            | #2                           | #3                          |
|------------|-------------------------------|------------------------------|-----------------------------|
| Qwen 7B    | SUBTLE_CONFLICT (1,123)       | REFUSAL_HALLUCINATION (960)  | SENTIMENT_MISREP. (250)     |
| Mistral 7B | SUBTLE_CONFLICT (2,728)       | SENTIMENT_MISREP. (373)      | REFUSAL_HALLUCINATION (253) |
| Aya23      | SUBTLE_CONFLICT (3,050)       | SENTIMENT_MISREP. (1,024)    | REFUSAL_HALLUCINATION (800) |
| GEITje     | SUBTLE_CONFLICT (3,699)       | BASELESS_INFO (2,444)        | SENTIMENT_MISREP. (1,273)   |

### Interpretation

- **SUBTLE_CONFLICT dominates across all models.** Paraphrase drift — responses that are "almost right" but shift meaning — is the primary failure mode in RAG. This is harder to prevent than outright fabrication because the model is partially grounded in the retrieved text.

- **Dutch models hallucinate at 2–3× the rate of English models.** GEITje (69.7%) and Aya23 (59.9%) are dramatically worse than GPT-4o-mini (30.9%) despite being purpose-built for Dutch. This suggests the bilingual interview domain is fundamentally harder for NL models — likely due to sparser Dutch pre-training data and fewer Dutch RAG supervision examples.

- **Query type failure patterns are model-specific and inverted.** Qwen fails at `speaker_attribution` (78%) but handles `temporal` well (13%). Mistral shows the exact opposite pattern (speaker_attribution: 3%, temporal: 2%, but sentiment: 66%). This implies different architectural failure modes — Qwen attends poorly to speaker markers in context; Mistral has a strong prior against refusal hallucinations on retrieval tasks.

- **GEITje generates BASELESS_INFO at alarming rates (2,444 instances).** This is the most severe hallucination type — complete fabrication. Dutch-specialised training on a smaller corpus appears to leave knowledge gaps that the model fills with invented content.

- **Specific_content queries are universally hard (82–99% hallucination).** These require verbatim retrieval of a precise moment in a conversation — a task that standard RAG generation cannot perform reliably because the model is inclined to paraphrase.

---

## 4. Experiment 2 — Model Scale

**Results:** `results/qwen14b/`

### What Changed from Experiment 1

The **only change**: model size doubled from **7B → 14B parameters** (Qwen2.5-7B → Qwen2.5-14B-Instruct). The RAG pipeline, prompt, retriever, and dataset are all identical.

This is a controlled ablation: does raw parameter count reduce hallucination, holding everything else constant?

### Result

| Model              | Rate   | Change                  |
|--------------------|--------|-------------------------|
| Qwen2.5-7B-Instruct | 41.8% | Baseline                |
| Qwen2.5-14B-Instruct | **16.4%** | **−25.4 pp (−61% relative)** |

### Interpretation

Scale provides a **dramatically larger benefit than any fine-tuning or decoding strategy tested**. The 14B model cuts hallucination by nearly two-thirds relative to 7B.

This result establishes that:
1. The 7B→14B jump matters substantially for faithfulness — the extra capacity improves instruction following, contextual grounding, and the model's ability to recognise when it lacks evidence.
2. The remaining **16.4% is a hard lower bound** for 14B-scale RAG on this task. It is the target for Experiments 4 and 5.
3. The 14B model primarily eliminates `REFUSAL_HALLUCINATION` — it finds answers rather than refusing. `SUBTLE_CONFLICT` and `SENTIMENT_MISREPRESENTATION` persist, indicating these are harder to fix with scale alone.

---

## 5. Experiment 3 — DPO Fine-tuning

**Scripts:** `experiments/06_finetuning/build_dpo_dataset.py`, `build_dpo_hard_negatives.py`, `run_dpo_*.py`, `run_qlora.py`
**Results:** `results/{model}_dpo/`, `results/geitje_sft/`

### What Changed from Experiment 1

Fine-tuned three models using **Direct Preference Optimisation (DPO)** on hallucination-corrected preference pairs:

- **Chosen response:** A GPT-4o-rewritten version of a hallucinated response (faithful, grounded)
- **Rejected response:** The original hallucinated RAG response
- **Training signal:** Push model distribution toward faithful over unfaithful responses

**Hard negatives** were added via `build_dpo_hard_negatives.py` — similar-looking but subtly incorrect responses placed as additional rejected candidates, to sharpen the preference boundary.

GEITje was first fine-tuned with **SFT (Supervised Fine-Tuning via QLora)** on faithful-only examples before DPO, since its raw 69.7% hallucination rate needed a baseline reset.

**DPO training config:**
- QLora rank-16, NF4 4-bit quantisation
- 3 epochs, lr=2e-4
- Training data: ~3,600 preference pairs from Experiment 1 annotations

### Results

| Model              | Before   | After DPO | Change                      |
|--------------------|----------|-----------|------------------------------|
| Qwen2.5-7B         | 41.8%    | **36.8%** | −5.0 pp ↓ (minor gain)      |
| Mistral-7B         | 33.1%    | **69.0%** | **+35.9 pp ↑ CATASTROPHIC** |
| GEITje-7B (SFT)    | 68.2%    | N/A       | SFT alone: no meaningful change |

### Interpretation

**DPO failed catastrophically for Mistral and was ineffective for GEITje.**

**Mistral inversion (33% → 69%)** — Catastrophic forgetting via preference collapse. DPO on ~3,600 domain-specific pairs caused Mistral to over-correct: it learned to "not hallucinate" by *refusing to answer*, generating vague disclaimers for nearly every query. The judge correctly penalises this as `REFUSAL_HALLUCINATION`. The model solved the training objective (avoid rejected responses) through the wrong mechanism (silence instead of faithful generation).

**Qwen minimal gain (−5 pp)** — DPO did not generalise from the training pairs to new transcripts. The model learned local corrections rather than a transferable faithfulness policy.

**GEITje SFT insufficient** — Even with all-faithful SFT examples, the model's underlying hallucination pattern does not shift meaningfully. The 68.2% rate persists, suggesting the problem is deeper than surface-level imitation.

**Conclusion:** DPO with a small domain dataset (~3,600 pairs) does not reliably reduce RAG hallucination. The risk of catastrophic degradation outweighs the potential gain for Mistral-class models. This result strongly motivates architectural interventions (Experiments 4 and 5) that do not require fine-tuning.

---

## 6. Experiment 4 — Context-Aware Decoding

**Scripts:** `experiments/fourth_results/run_fourth_results.py`, `cad_decoding.py`, `prompts.py`
**Results:** `results/fourth_results/{model}/`

### What Changed from Experiment 1

Two simultaneous changes:
1. An **evidence-citation prompt** requiring the model to inline-cite every claim
2. **Context-Aware Decoding (CAD)** — a modified decoding algorithm

### Evidence-Citation Prompt

```
System:
  You are a research assistant answering questions about interview transcripts.
  For EVERY factual claim you make, immediately follow it with a verbatim citation
  in this exact format:  [CITE: "exact quote from transcript"]

  RULES:
  - The cited text must be copied word-for-word from the transcript.
  - Do NOT make any claim you cannot directly cite.
  - If the answer is not present: state "The transcript does not contain this information."

Context:
  {retrieved_chunks}

Question:
  {query}
```

### Context-Aware Decoding (CAD)

CAD (Shi et al., NeurIPS 2023, arXiv:2305.14739) modifies the decoding distribution by **contrasting context-conditioned vs context-free generation**:

```
p_CAD(y_t | y<t) = softmax[
    (1 + α) · logit(y_t | context, query, y<t)
  −       α · logit(y_t | query, y<t)
]
```

Tokens that the model would generate *without* the context are down-weighted. This forces the model to rely more heavily on the retrieved evidence. The α parameter controls the strength of the contrastive effect.

**α values per model (tuned to model sensitivity):**

| Model        | α   | Rationale                                         |
|-------------|-----|---------------------------------------------------|
| Qwen2.5-7B  | 1.0 | Paper's optimal for conflict-QA tasks             |
| Aya-23-8B   | 0.7 | Reduced — multilingual models are more sensitive  |
| GEITje-7B   | 0.5 | Minimal — Dutch model fragile to high α           |
| Mistral-7B  | 1.0 | Same as Qwen (mistake — see failure below)        |

### Results

| Model        | Baseline | CAD + Cite-Prompt | Change                      |
|-------------|----------|-------------------|-----------------------------|
| Qwen2.5-7B  | 41.8%    | Pending ~Apr 5    | —                           |
| Aya-23-8B   | 59.9%    | Pending ~Apr 6    | —                           |
| GEITje-7B   | 69.7%    | Pending ~Apr 6    | —                           |
| Mistral-7B  | 33.1%    | **99.4%**         | **+66.3 pp ↑ CATASTROPHIC** |

### Mistral CAD Failure Analysis

Mistral's 99.4% hallucination with CAD is caused by a **single type dominating the entire result**:
- `REFUSAL_HALLUCINATION`: 3,598 / 3,600 items (99.9% of all hallucinations)

The α=1.0 value was far too aggressive for Mistral. The contrastive subtraction removed too much probability mass from context-free tokens, causing the model's output distribution to degenerate. Mistral could no longer produce fluent responses and defaulted to outputting "I cannot find this information in the transcript" for virtually every query — which is correctly flagged as REFUSAL_HALLUCINATION, because the answers *do* exist in the transcript.

**Key insight:** CAD's α must be calibrated per model. What is the paper's optimal for Qwen becomes a catastrophic collapse regime for Mistral. The lower α values for Aya23 (0.7) and GEITje (0.5) are expected to avoid this collapse.

### Why CAD is Theoretically Motivated

The dominant hallucination type from Experiment 1 is `SUBTLE_CONFLICT` — the model generates text that is "mostly right" but paraphrases or slightly modifies retrieved content. CAD specifically targets this by amplifying context-dependent tokens at each generation step, making it harder for the model to drift from what the retrieved context says.

---

## 7. Experiment 5 — Three Faithfulness Approaches

**Scripts:** `experiments/fifth_results/run_fifth_results.py`, `nli_verifier.py`, `reprocess_dutch_nli.py`
**Results:** `results/fifth_results/{approach}/{model}/`

Experiment 5 tests three fundamentally different architectural strategies for faithfulness, applied to the same base models without fine-tuning.

---

### Approach A — Extractive RAG (Verbatim Span Extraction)

#### Core Idea

Instead of asking the model to *generate* an answer, instruct it to *locate and copy* the exact verbatim span from the transcript that answers the query. Generation = retrieval, not synthesis.

#### v1 Prompt (Pure Extraction)

```
System:
  You are a transcript span extractor. Your ONLY job is to find and copy
  verbatim text from the interview transcript.

  HARD RULES:
  - Copy word-for-word. Do NOT paraphrase.
  - Do NOT add context, explanation, or your own words.
  - If the answer is not present: output exactly:  NO RELEVANT INFORMATION

  INTERVIEW TRANSCRIPT:
  {context}
```

#### v1 Failure: 46.9% Hallucination for Qwen (Worse Than Baseline)

Root cause analysis revealed three structural mismatches between query types and verbatim extraction:

| Query Type        | v1 Failure Rate | Root Cause                                                                        |
|------------------|-----------------|-----------------------------------------------------------------------------------|
| `factual_summary` | **100%**        | Synthesis cannot be answered by a single span → "NO RELEVANT INFORMATION" → REFUSAL_HALLUCINATION |
| `sentiment`       | **100%**        | No verbatim span expresses raw sentiment → same refusal pattern                  |
| `specific_content`| **55.6%**       | Temporal-relational queries: model extracts the anchor quote, not the following turn |

Dominant types: SPEAKER_MISATTRIBUTION (11), TEMPORAL_CONFUSION (10) for `specific_content`.

#### v2 Fix: Query-Type-Aware Prompt Routing

Three different system prompts are selected based on the query type:

```python
_SYNTHESIS_TYPES  = {"factual_summary", "sentiment"}
_RELATIONAL_TYPES = {"specific_content"}

def _get_extract_system(query_type: str, lang: str) -> str:
    if query_type in _SYNTHESIS_TYPES:
        return _SYNTH_SYSTEM_NL if lang == "nl" else _SYNTH_SYSTEM_EN
    if query_type in _RELATIONAL_TYPES:
        return _RELATIONAL_SYSTEM_NL if lang == "nl" else _RELATIONAL_SYSTEM_EN
    return _EXTRACT_SYSTEM_NL if lang == "nl" else _EXTRACT_SYSTEM_EN
```

**Synthesis prompt** (for `factual_summary`, `sentiment`):
```
Every factual claim or sentiment observation MUST be followed immediately by
a verbatim citation:  [CITE: "exact text from transcript"]
Do NOT make any claim that cannot be supported by a direct verbatim citation.
```

**Relational prompt** (for `specific_content`):
```
STEP 1 — Find the anchor quote mentioned in the question.
STEP 2 — Copy the turn that DIRECTLY FOLLOWS (or precedes) that anchor.

OUTPUT FORMAT:
ANCHOR: [Speaker]: "[exact anchor quote]"
ANSWER: [Speaker]: "[exact verbatim text of the turn that follows/precedes]"

HARD RULE: The ANSWER must be a DIFFERENT turn from the ANCHOR.
```

#### Results

| Model       | Rate    | vs Baseline | Notes                      |
|------------|---------|------------|----------------------------|
| Aya-23-8B  | **39.7%** | −20.2 pp ↓ | Best result so far for Aya |
| Qwen 7B v1 | 46.9%   | +5.1 pp ↑  | v1 only — v2 pending       |
| Qwen 7B v2 | Pending | —           | Running (Apr 5 ETA)        |
| GEITje-7B  | Pending | —           | Running (Apr 5 ETA)        |
| Mistral-7B | Stuck   | —           | Together API 503 outage    |

**Aya23 extractive breakdown (v2):**

| Query Type            | Rate |
|-----------------------|------|
| `temporal`            | 55%  |
| `sentiment`           | 49%  |
| `participant_content` | 43%  |
| `factual_summary`     | 41%  |
| `specific_content`    | 34%  |
| `speaker_attribution` | **9%** ← near-perfect |

Extractive is highly effective for `speaker_attribution` (9%) — the model copies the relevant speaker's words verbatim. It still struggles with `temporal` and `sentiment` even with routing.

---

### Approach B — NLI Filter (Post-Generation Faithfulness Filtering)

**Script:** `experiments/fifth_results/nli_verifier.py`

#### Core Idea

Generate a response normally, then **filter out unfaithful claims** using a Natural Language Inference (NLI) model. Only claims entailed by the retrieved context are kept.

```
For each claim C in the response:
  premise    = retrieved transcript chunk
  hypothesis = claim C

  NLI_score = P(entailment | premise, hypothesis)

  IF score ≥ 0.5 → keep claim (context-grounded)
  IF score < 0.5 → discard claim (not supported)

Final response = " ".join(kept_claims)
```

#### Critical Bug: English-Only NLI Model

**v1 (broken):** `cross-encoder/nli-deberta-v3-small`

This model is English-only. Applied to Dutch transcripts, it produced near-zero entailment scores regardless of actual faithfulness.

**Impact:**
- English cite-keep rate: 51.8% (reasonable)
- Dutch cite-keep rate: 23.6% (collapsed — almost nothing kept)
- 39.2% of Dutch responses reduced to `"NO VERIFIED CLAIMS"` (empty output)
- 41% of the full dataset was effectively invalidated

**Fix (v2):** Replaced with `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- 750 MB multilingual NLI model (100+ languages including Dutch)
- Trained on XNLI + 2 million multilingual NLI pairs
- Dynamic entailment label index (mDeBERTa uses label index 0, not 1 as in DeBERTa — detected automatically via `model.config.id2label`)

**Additional code fix:** The `_batch_nli_scores()` function was refactored to infer its device from `next(model.parameters()).device` instead of accepting a `device` argument. The old call `_batch_nli_scores(..., device=device)` raised `TypeError` because the new signature removed that parameter.

**Dutch recovery after fix:** Dutch cite-keep rate recovered from 23.6% → ~61% on first 150 items. Annotation of the full fixed dataset is running (job 1378202).

---

### Approach C — GPT-4o + NLI (Citation-Forced Generation + Verification)

**Script:** `experiments/fifth_results/run_fifth_results.py` → `run_gpt4o_nli()`

#### Core Idea

A two-stage faithfulness pipeline using GPT-4o as the generator with mandatory inline citations, followed by NLI verification of each claim:

**Stage 1 — Citation-forced generation (GPT-4o):**
```
System:
  Every factual claim MUST be followed immediately by a verbatim citation:
  [CITE: "exact text from transcript"]
  Do NOT make any claim that cannot be supported by a direct verbatim citation.
  Maximum 150 words.
```

**Stage 2 — NLI verification:**
```
Parse response → extract (claim, citation) pairs
For each pair:
  NLI(premise=citation, hypothesis=claim)
  IF entailment_score ≥ 0.5 → verified_claims.append(claim)

Final response = " ".join(verified_claims)
IF no verified claims → "NO VERIFIED CLAIMS"
```

The NLI model acts as a faithfulness gate: even if GPT-4o generates a citation, the NLI verifier checks whether the claim actually follows from that citation.

#### Dutch NLI Issue (Same as Approach B)

The same English-only DeBERTa collapse affected this approach. 39.2% of Dutch items were reduced to "NO VERIFIED CLAIMS" before the mDeBERTa fix.

**Efficient fix:** Because the `raw_response` field is stored in `01_responses.json`, the `reprocess_dutch_nli.py` script can **re-run NLI filtering without new GPT-4o API calls** — reusing existing GPT-4o responses, just re-scoring them with mDeBERTa.

---

## 8. Hallucination Detection & RQ2

**Scripts:** `experiments/03_rq2/experiment_rq2.py`, `experiments/07_validation/detector_eval.py`

### RQ2: Can Automatic Detectors Replace the GPT-4o Judge?

Evaluated four automatic hallucination detectors against GPT-4o ground truth labels on Qwen2.5-14B baseline responses (n=3,600 — the dataset with the most reliable annotations).

| Detector    | Precision | Recall | F1      | Description                                      |
|-------------|-----------|--------|---------|--------------------------------------------------|
| **RAGAS**   | 0.820     | 0.185  | **0.302** | Token-level faithfulness scoring                |
| **MiniCheck**| 0.170    | 0.864  | **0.284** | Claim-level entailment checking                 |
| SelfCheck   | 0.081     | 0.167  | 0.111   | Sampling-based consistency checking             |
| AlignScore  | 0.027     | 0.067  | 0.043   | N-gram + semantic alignment                     |

### Interpretation

**No detector achieves F1 > 0.41.** This is a strong negative result:

- **RAGAS** has excellent precision (0.820) but extremely low recall (0.185). It catches only the most obvious hallucinations — missing 81.5% of true positives. Useful for flagging high-confidence failures but cannot be used as a judge replacement.

- **MiniCheck** has strong recall (0.864) but precision of 0.170 — meaning 83% of its hallucination flags are false alarms. Would cause massive over-rejection of faithful responses in production.

- **SelfCheck and AlignScore** are ineffective for this task. The nuanced, transcript-grounded hallucination types (SUBTLE_CONFLICT, SENTIMENT_MISREPRESENTATION) are invisible to these generic detectors.

**Root cause:** The 7-type taxonomy with severity levels captures domain-specific, fine-grained failures that generic detectors were not designed to detect. The bilingual (NL/EN) setting also degrades English-centric detectors. A dedicated judge (GPT-4o or local Qwen2.5-32B) is necessary for reliable annotation.

---

## 9. Ablation & Hyperparameter Studies

**Scripts:** `experiments/05_ablation/`, `experiments/06_hyperparameter/`
**Results:** `RAG_THESIS/output/hyperparameter_summary.json`, `sweep_consolidated_table.csv`

### 9.1 No-RAG Baseline

**Script:** `experiments/05_ablation/run_no_rag_baseline.py`

Answers all queries with **no retrieved context** — the model uses only its parametric knowledge. Establishes what fraction of hallucination is inherent to the model (independent of retrieval) vs. introduced by the RAG pipeline.

### 9.2 Chain-of-Thought Ablation

**Script:** `experiments/05_ablation/run_ablation_cot.py`

Adds explicit "think step-by-step" reasoning before answering. Tests whether structured reasoning reduces hallucination or creates additional opportunities for drift.

### 9.3 Temperature Analysis

**Script:** `experiments/06_hyperparameter/temperature_analysis.py`

Systematically varies temperature across {0.0, 0.3, 0.5, 0.7, 1.0}. Lower temperature = more deterministic generation. Hypothesis: does greedy decoding (T=0) reduce hallucination rates?

### 9.4 Top-K × Chunk-Size Grid Search

**Scripts:** `experiments/06_hyperparameter/top_k_chunk_analysis.py`, `run_ablation_sweep.py`

| Parameter    | Values Tested       | Default Used |
|-------------|---------------------|--------------|
| `top_k`     | 1, 3, 5, 7, 10      | 5            |
| `chunk_size`| 256, 512, 1024 tok  | 512          |
| `temperature`| 0.0–1.0            | 0.0          |

Results consolidated in `analyze_sweep_results.py` → saved to `RAG_THESIS/output/`.

---

## 10. Validation & Inter-Annotator Agreement

**Scripts:** `experiments/07_validation/`

### 10.1 Human Kappa Study

**Script:** `export_annotation_csv.py` → generates `manual_validation_30.csv`

30 items stratified across models and query types are exported for independent human annotation. The human annotator labels:
- `YOUR_label`: HALLUCINATED or FAITHFUL
- `YOUR_types`: comma-separated types from the 7-type taxonomy

`calculate_kappa.py` computes Cohen's κ between GPT-4o judge and human labels.

**Status:** CSV exported. Pending human annotation.

### 10.2 Retrieval Grounding Evaluation

**Script:** `retrieval_grounding_eval.py`

Evaluates whether the retrieved chunks actually contain evidence for the model's answer. Separates **retrieval failure** (wrong chunks returned) from **generation failure** (right chunks, wrong answer).

Uses BGE-M3 to compute cosine similarity between model claims and retrieved context. If the answer cannot be grounded in retrieved chunks, the hallucination is attributable to retrieval failure rather than generation drift.

### 10.3 Local Annotation Judge (Free Alternative to GPT-4o)

**Script:** `experiments/local_judge.py`

Replaces the GPT-4o judge with locally-run **Qwen2.5-32B-Instruct** using 4-bit NF4 quantisation (~16 GB VRAM). Auto-selects the best available model:

```
32B (4-bit, ~16 GB)  →  14B (4-bit, ~8 GB)  →  7B (fp16, ~14 GB)
```

All three fit on L4-24G nodes. No API cost. ~2–6h per 3,600-item annotation run.

```bash
# Annotate a specific results directory
python experiments/local_judge.py --results-dir results/fifth_results/extractive/qwen

# Annotate all pending experiment directories at once
python experiments/local_judge.py --all

# Re-annotate quota-failed items from API outage
python experiments/local_judge.py --results-dir ... --retry-failed

# Force a specific model size
python experiments/local_judge.py --all --model 32b
```

**Estimated quality vs GPT-4o:**

| Judge           | Relative Quality | Cost per 3,600 items |
|----------------|-----------------|----------------------|
| GPT-4o (API)   | 100% (reference) | ~€22                |
| Qwen2.5-32B local | ~80%         | €0                  |
| Qwen2.5-14B local | ~65%         | €0                  |
| Qwen2.5-7B local  | ~55%         | €0                  |

---

## 11. Key Findings & Interpretation

### Master Results Table

| Phase | Model / Approach         | Rate      | Δ vs Model Baseline |
|-------|--------------------------|-----------|---------------------|
| **E1 Baseline** | GPT-4o-mini          | 30.9%     | —               |
| **E1 Baseline** | Mistral-7B           | 33.1%     | —               |
| **E1 Baseline** | Qwen2.5-7B           | 41.8%     | —               |
| **E1 Baseline** | Aya-23-8B            | 59.9%     | —               |
| **E1 Baseline** | GEITje-7B            | 69.7%     | —               |
| **E2 Scale**    | Qwen2.5-14B          | **16.4%** | −25.4 pp ↓      |
| **E3 DPO**      | Qwen2.5-7B-DPO       | 36.8%     | −5.0 pp ↓       |
| **E3 DPO**      | Mistral-7B-DPO       | 69.0%     | **+35.9 pp ↑**  |
| **E3 SFT**      | GEITje-7B-SFT        | 68.2%     | −1.5 pp ↓       |
| **E4 CAD**      | Mistral-7B (α=1.0)   | 99.4%     | **+66.3 pp ↑**  |
| **E4 CAD**      | Qwen/Aya/GEITje      | Pending   | —               |
| **E5 Extr.**    | Aya-23-8B            | **39.7%** | −20.2 pp ↓      |
| **E5 Extr.**    | Qwen/GEITje          | Pending   | —               |
| **E5 GPT4o+NLI**| All models           | Running   | —               |

### Finding 1: Scale Is the Most Effective Single Intervention

Doubling model size (7B → 14B) reduced Qwen's hallucination by 61% relative — more than any fine-tuning or decoding strategy. This suggests 7B models are capacity-limited for faithful RAG on complex bilingual transcripts.

### Finding 2: SUBTLE_CONFLICT is the Dominant and Hardest Failure Mode

Across all models and approaches, paraphrase drift (slight meaning changes, not outright fabrication) accounts for the majority of hallucinations. It is difficult to prevent because the model is *partially right* — grounded in retrieved text but shifting nuance. Extractive and NLI approaches target this specifically.

### Finding 3: Dutch Transcripts Are Structurally Harder

Even Dutch-specialised models (GEITje: 69.7%, Aya23: 59.9%) hallucinate at roughly 2× the rate of English-first models (GPT-4o-mini: 30.9%, Mistral: 33.1%). The Dutch NLI collapse (English-only DeBERTa) further illustrates that existing NLP tooling assumes English as the default, creating compounding disadvantages for NL.

### Finding 4: Fine-tuning Backfires Without Sufficient Domain Data

DPO on ~3,600 pairs caused Mistral to increase from 33.1% → 69.0% via preference collapse. The model learns to refuse rather than to be faithful. Fine-tuning for faithfulness requires larger, more diverse preference datasets, or approaches like RLHF with a reward model that penalises both hallucination AND refusal.

### Finding 5: CAD's Alpha is Model-Critical

Mistral at α=1.0 shows catastrophic collapse (99.4%). The same α for Qwen is the paper's recommended optimum. Model sensitivity to CAD is not predictable from model size or family alone — it must be calibrated empirically.

### Finding 6: Extractive RAG Significantly Reduces Hallucination

The v2 extractive approach (query-type-aware routing) reduces Aya23 from 59.9% → 39.7% — the largest reduction seen for that model across all approaches. For tasks reducible to verbatim retrieval (`speaker_attribution`: 9%), performance is near-perfect. The remaining error is in synthesis-required query types (`sentiment`, `temporal`).

### Finding 7: Automatic Detectors Are Insufficient

Max F1 = 0.302 (RAGAS) — no existing automatic detector can reliably identify the nuanced, domain-specific hallucination types in this setting. A dedicated judge (GPT-4o or large local model) is necessary for reliable annotation.

### Research Contributions

1. Systematic evaluation of 6 LLMs on bilingual (NL/EN) interview transcript RAG across 5 progressive phases
2. Discovery and documentation of the Dutch NLI model failure (English-only DeBERTa collapse on Dutch)
3. Evidence that DPO worsens hallucination when domain data is insufficient
4. Demonstration that CAD is highly model-sensitive (α=1.0 causes collapse in Mistral)
5. Query-type-aware extractive routing as a practical, training-free hallucination mitigation strategy
6. First evaluation of RAGAS / MiniCheck / SelfCheck / AlignScore on Dutch interview RAG
7. Local Qwen2.5-32B as a cost-free annotation judge framework

---

## 12. Reproducibility & Job Scripts

### Environment

```bash
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load OpenSSL/1.1
source ~/thesis_venv/bin/activate

export HF_HOME="/zfsstore/user/s4238206/hf_cache"
```

### GPU Partitions

| Partition       | VRAM  | Used For                           |
|-----------------|-------|------------------------------------|
| `gpu-l4-24g`    | 24 GB | All generation experiments (7B–32B)|
| `gpu-a100-80g`  | 80 GB | Mixtral inference                  |
| `cpu-zen4`      | —     | API-only jobs (annotation, ablation)|

### Experiment Execution Order

```bash
# Phase 0 — Data
sbatch jobs/run_data_generation.sh

# Phase 1 — Baseline RAG (all models)
sbatch jobs/run_all_models.sh

# Phase 2 — Scale (already embedded in baseline as qwen14b)

# Phase 3 — DPO
sbatch jobs/run_dpo_qwen.sh
sbatch jobs/run_dpo_mistral.sh
sbatch jobs/run_geitje_sft.sh

# Phase 4 — CAD
sbatch jobs/run_fourth_qwen.sh
sbatch jobs/run_fourth_aya23.sh
sbatch jobs/run_fourth_geitje.sh

# Phase 5 — Faithfulness approaches
sbatch jobs/run_fifth_extractive.sh        # Approach A
# (Approach B: NLI filter runs inside fifth pipeline)
sbatch jobs/run_fifth_gpt4o_nli.sh        # Approach C

# Fix Dutch NLI (Approaches B & C)
sbatch jobs/run_fix_dutch_nli.sh          # re-runs with mDeBERTa

# Local annotation (free, no API)
sbatch jobs/download_qwen32b.sh           # one-time: download 32B judge
sbatch jobs/run_local_annotation.sh       # annotate all results

# Analysis & validation
sbatch jobs/run_validation.sh
sbatch jobs/run_ablation_hyperparameter.sh

# Thesis PDF report
python experiments/generate_thesis_report.py
```

### Environment Variables (`.env`)

```
OPENAI_API_KEY=sk-...        # GPT-4o judge (or use local_judge.py for free)
HF_TOKEN=hf_...              # HuggingFace model access
TOGETHER_API_KEY=...         # Together AI (Mixtral API inference)
```

---

*Leiden University · LIACS · Master's Thesis 2025–2026*
*Sai Krishna Reddy Mulakkayala (s4238206)*
*Supervisors: [add supervisors]*
*Last updated: April 2026*
