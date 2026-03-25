# Pipeline Connections — Complete File Dependency Map

Every Python file in this project, what it reads, what it writes, and why it is a valid IEEE contribution.

---

## Data Flow (top to bottom)

```
DATA SOURCES
────────────────────────────────────────────────────────────────────
data/real/supabase_responses.csv           (real interview transcripts)
data/synthetic/synthetic_interviews.csv    (English synthetic)
data/synthetic/supbase_english_synthetic.csv
data/synthetic/supbase_dutch_synthetic.csv  (Dutch synthetic — essential for GEITje)
                        │
                        ▼
═══════════════════════════════════════════════════════════════════
STAGE 1: DATA LOADING & RAG GENERATION
═══════════════════════════════════════════════════════════════════
experiments/01_pipeline/data_loader.py
  READS:   the 4 CSV files above
  WRITES:  List[InterviewDict] (in memory)
  CONTRIBUTION: Unified multilingual data loader (real + synthetic, EN + NL)

experiments/01_pipeline/model_registry.py         ← NEW
  READS:   (nothing — pure config)
  WRITES:  MODELS dict (in memory)
  CONTRIBUTION: Reproducible 6-model config; anyone can replicate exact setup

utils/rag_configs.py
  READS:   (nothing — pure config)
  WRITES:  CONFIGS, HYPERPARAMETER_GRID, DEFAULTS (in memory)
  CONTRIBUTION: Systematic hyperparameter space definition

utils/embeddings_retriever.py                     ← UPDATED: chunk_overlap + MMR
  READS:   interview utterances (in memory)
  WRITES:  chunk embeddings (in memory cache)
  KEY FUNCTIONS:
    chunk_utterances(utterances, chunk_size, chunk_overlap=0)
    retrieve_chunks(query, interview, chunk_size, top_k, client,
                    chunk_overlap=0, retrieval_strategy="top_k",
                    similarity_threshold=0.0)
    _retrieve_mmr(query_emb, chunks, embeddings, top_k, lambda_param=0.5)
  CONTRIBUTION: MMR retrieval + chunk overlap — novel for interview domain

experiments/01_pipeline/rag_pipeline.py           ← UPDATED: HF models
  READS:   model_registry.py (model configs)
           embeddings_retriever.py (chunk retrieval)
  WRITES:  results/{model}/01_rag_responses.json
  KEY FUNCTIONS:
    _generate_hf(messages, model_key, temperature, max_new_tokens, top_p, repetition_penalty)
    generate_response(query, context, model, temperature, system_prompt_style, ...)
    generate_samples(query, context, n_samples, model, ...)
  CONTRIBUTION: Unified API + HuggingFace generation; enables GEITje + Aya-23

experiments/01_pipeline/run_all.py                ← UPDATED: 6 models + new steps
  READS:   data CSVs, per-model results
  WRITES:  orchestrates all steps
  CONTRIBUTION: Single command to run full 6-model pipeline
  USAGE:
    python experiments/01_pipeline/run_all.py --models geitje,aya23 --steps all

═══════════════════════════════════════════════════════════════════
STAGE 2: RQ1 — HALLUCINATION TAXONOMY
═══════════════════════════════════════════════════════════════════
experiments/02_rq1/experiment_rq1.py
  READS:   results/{model}/01_rag_responses.json
  WRITES:  results/{model}/02_rq1_annotations.json    (RAGTruth labels)
           results/{model}/03_rq1_huang_taxonomy.json (Huang et al. labels)
           results/{model}/04_rq1_diahalu_eval.json   (DiaHaLu labels)
  CONTRIBUTION: C2 — First multi-taxonomy comparison in interview RAG domain
                C3 — Discovery of REFUSAL_HALLUCINATION pattern

═══════════════════════════════════════════════════════════════════
STAGE 3: RQ2 — DETECTOR EVALUATION
═══════════════════════════════════════════════════════════════════
experiments/03_rq2/experiment_rq2.py
  READS:   results/{model}/01_rag_responses.json
  WRITES:  results/{model}/05_rq2_selfcheck.json   (SelfCheckGPT)
           results/{model}/06_rq2_minicheck.json   (MiniCheck NLI)
           results/{model}/07_rq2_alignscore.json  (AlignScore)
           results/{model}/08_rq2_ragas.json       (RAGAS faithfulness)
  CONTRIBUTION: C5 — First systematic detector comparison in interview domain
                     SelfCheckGPT failure (F1=0) is a key finding

═══════════════════════════════════════════════════════════════════
STAGE 4: MERGE ALL RESULTS
═══════════════════════════════════════════════════════════════════
utils/build_unified_results.py                    ← UPDATED: 6 models
  READS:   results/{model}/02–08 JSONs for ALL 6 models
  WRITES:  data/unified_results.jsonl
           (one JSON record per response, all fields merged)
  CONTRIBUTION: Enables cross-model statistical comparison

═══════════════════════════════════════════════════════════════════
STAGE 5: ADVANCED EXPERIMENTS (parallel, all read unified JSONL)
═══════════════════════════════════════════════════════════════════

experiments/04_advanced/ensemble_detector.py
  READS:   results/{model}/05–08 RQ2 JSONs
           RAG_THESIS/output/manual_validation_200.csv (YOUR_label column)
  WRITES:  RAG_THESIS/output/ensemble_results.json
           RAG_THESIS/output/ensemble_comparison_chart.png
  CONTRIBUTION: C5 — Learned ensemble (LOO-CV F1=0.871) beats all individuals
  KEY FINDING: MiniCheck contributes most (highest β weight)

experiments/04_advanced/hallucination_predictor.py
  READS:   data/unified_results.jsonl
  WRITES:  RAG_THESIS/output/predictor_results.json
  CONTRIBUTION: C6 — Predictive detection before generation (novel approach)

experiments/04_advanced/correction_loop.py
  READS:   results/{model}/01_rag_responses.json
  WRITES:  RAG_THESIS/output/corrected_responses.json
  CONTRIBUTION: C6 — Auto-correction demonstrates practical fix value

experiments/04_advanced/retrieval_correlation.py
  READS:   data/unified_results.jsonl
  WRITES:  RAG_THESIS/output/retrieval_correlation.json
  CONTRIBUTION: Links retrieval quality to hallucination type distribution

experiments/05_ablation/run_no_rag_baseline.py
  READS:   data/ CSV files
  WRITES:  RAG_THESIS/output/no_rag_baseline.json
  CONTRIBUTION: Proves RAG reduces hallucinations — required by reviewers

experiments/05_ablation/run_ablation_cot.py
  READS:   data/ CSV files
  WRITES:  RAG_THESIS/output/cot_ablation.json
  CONTRIBUTION: Tests Chain-of-Thought effect on hallucination rates

experiments/06_hyperparameter/run_ablation_sweep.py
  READS:   data/ CSV files
  WRITES:  RAG_THESIS/output/hyperparameter_sweep_results.json
  CONTRIBUTION: Systematic 10-config ablation across prompt+retrieval

experiments/06_hyperparameter/chunk_overlap_analysis.py        ← NEW
  READS:   data/ CSV files
  WRITES:  RAG_THESIS/output/chunk_overlap_results.json
           RAG_THESIS/output/chunk_overlap_chart.png
  CONTRIBUTION: MOST NOVEL: chunk_overlap × MMR interaction study
  EXPECTED FINDING: overlap=100 + MMR reduces CONTEXT_FABRICATION by ~15%

experiments/06_hyperparameter/temperature_analysis.py
  READS:   data/ CSV files
  WRITES:  RAG_THESIS/output/temperature_results.json
  CONTRIBUTION: Temperature vs hallucination rate trade-off

experiments/06_hyperparameter/top_k_chunk_analysis.py
  READS:   data/ CSV files
  WRITES:  RAG_THESIS/output/topk_chunk_results.json
  CONTRIBUTION: Chunk size × top_k interaction

experiments/06_hyperparameter/threshold_optimizer.py
  READS:   data/unified_results.jsonl
  WRITES:  RAG_THESIS/output/threshold_results.json
  CONTRIBUTION: Optimal detection threshold per detector

experiments/06_hyperparameter/sampling_analysis.py
  READS:   data/unified_results.jsonl
  WRITES:  RAG_THESIS/output/sampling_results.json
  CONTRIBUTION: Effect of SelfCheckGPT sample count on F1

experiments/07_validation/export_annotation_csv.py
  READS:   results/gpt-4o-mini/02_rq1_annotations.json
  WRITES:  RAG_THESIS/output/manual_validation_200.csv
  CONTRIBUTION: Exports 198 stratified samples for human annotation

experiments/07_validation/calculate_kappa.py
  READS:   RAG_THESIS/output/manual_validation_200.csv (YOUR_label column)
  WRITES:  RAG_THESIS/output/validation_results.json
  CONTRIBUTION: Cohen's Kappa ≥ 0.61 validates GPT-4o-mini judge (IEEE req.)

experiments/07_validation/detector_eval.py
  READS:   data/unified_results.jsonl
           RAG_THESIS/output/manual_validation_200.csv
  WRITES:  RAG_THESIS/output/detector_eval_results.json
  CONTRIBUTION: Detector performance against human ground truth

═══════════════════════════════════════════════════════════════════
STAGE 6: STATISTICAL ANALYSIS
═══════════════════════════════════════════════════════════════════
utils/statistics_utils.py
  READS:   data/unified_results.jsonl
           RAG_THESIS/output/manual_validation_200.csv (optional)
  WRITES:  RAG_THESIS/output/statistics_report.json
  KEY FUNCTIONS:
    wilson_ci(p, n, z=1.96)          → 95% CI for proportions
    mcnemar_test(labels_a, labels_b) → pairwise model comparison
    compare_models(unified_jsonl)    → all 6-model pairwise McNemar
    detector_confidence_intervals()  → P/R/F1 with CIs per detector
  CONTRIBUTION: Statistical rigour required for IEEE acceptance
  KEY RESULTS (3-model prior run):
    GPT-4o-mini vs Mistral: χ²=15.614, p=0.0001 (SIGNIFICANT)
    Mistral vs Qwen:        χ²=31.161, p=0.0000 (SIGNIFICANT)

utils/compare_models.py
  READS:   results/{model}/ directories
  WRITES:  results/comparison/ report files
  CONTRIBUTION: Cross-model comparison table for IEEE paper

═══════════════════════════════════════════════════════════════════
STAGE 7: PDF REPORT GENERATION
═══════════════════════════════════════════════════════════════════
reports/generate_thesis_report.py
  READS:   all JSON result files
           statistics_report.json, ensemble_results.json, etc.
  WRITES:  results/thesis_ieee_report.pdf
  CONTRIBUTION: IEEE-structured 10-section paper

reports/generate_companion_guide.py
  READS:   all JSON result files
  WRITES:  results/thesis_companion_guide.pdf
  CONTRIBUTION: 25-page practical guide (run order, annotation tutorial,
                Q&A for supervisor, all numbers explained)
```

---

## Key File Interactions (who calls whom)

```
run_all.py
  ├── data_loader.py       (merge_csv_sources, prepare_all_samples)
  ├── rag_pipeline.py      (generate_all_responses)
  │     ├── model_registry.py   (_HF_MODEL_IDS)
  │     └── embeddings_retriever.py (retrieve_chunks)
  ├── experiment_rq1.py    (run_all_rq1)
  ├── experiment_rq2.py    (run_all_rq2)
  ├── compare_models.py    (generate_comparison_report)
  ├── build_unified_results.py (build)
  └── statistics_utils.py  (compare_models, detector_confidence_intervals)

ensemble_detector.py
  └── reads results/{model}/05–08 JSONs directly (no imports from pipeline)

chunk_overlap_analysis.py
  ├── data_loader.py
  ├── rag_pipeline.py (generate_response)
  ├── embeddings_retriever.py (retrieve_chunks ← uses chunk_overlap + MMR)
  └── experiment_rq1.py (classify_response — judge)
```

---

## What Each New Model Adds

### GEITje-7B-Ultra (`results/geitje/`)
- **Thesis claim:** "First Dutch-specific RAG hallucination study"
- **Expected finding:** Lower hallucination rate on Dutch interview queries vs Qwen/Llama
- **Comparison:** GEITje vs Aya-23 on Dutch queries = Dutch-specific vs multilingual
- **Section:** C1 (Dutch contribution) + C4 (cross-lingual)

### Aya-23-8B (`results/aya23/`)
- **Thesis claim:** "Multilingual Dutch baseline for cross-lingual comparison"
- **Expected finding:** Better Dutch than Qwen/Llama (23-language training), worse than GEITje
- **Comparison:** Aya-23 vs GEITje = best Dutch question in the paper
- **Section:** C4 (cross-lingual contribution)

### Llama-3.1-8B (`results/llama/`)
- **Thesis claim:** "Reproducibility baseline — most cited open-source LLM"
- **Expected finding:** Moderate hallucination rate; 128K context a major advantage
- **Comparison:** Llama vs Qwen (both ~8B, different training data)
- **Section:** Section 4 (model comparison)

---

## Hyperparameter Contribution Map

| Parameter | Where Swept | Expected Effect | Section |
|-----------|------------|-----------------|---------|
| `chunk_overlap` [0,50,100,200] | `06_hyperparameter/chunk_overlap_analysis.py` | Reduces CONTEXT_FABRICATION | 6.1 |
| `retrieval_strategy` [top_k, mmr] | `06_hyperparameter/chunk_overlap_analysis.py` | MMR reduces redundant chunks | 6.1 |
| `system_prompt_style` [detailed, brief, strict_grounding] | `06_hyperparameter/run_ablation_sweep.py` | strict_grounding reduces BASELESS_INFO | 6.2 |
| `temperature` [0.0, 0.3, 0.7, 1.0] | `06_hyperparameter/temperature_analysis.py` | Lower T = fewer hallucinations | 6.3 |
| `chunk_size` [256, 512, 1024] | `06_hyperparameter/top_k_chunk_analysis.py` | 512 optimal for interview chunks | 6.4 |
| `top_k` [3, 5, 10] | `06_hyperparameter/top_k_chunk_analysis.py` | top_k=5 best F1/hallucination trade-off | 6.4 |
| `selfcheck_n_samples` [3, 5, 10] | `06_hyperparameter/sampling_analysis.py` | n=5 sufficient; n=10 marginal gain | 6.5 |
| `repetition_penalty` [1.0, 1.1, 1.2] | `06_hyperparameter/run_ablation_sweep.py` | 1.1 reduces repetitive hallucinations (HF) | 6.6 |
