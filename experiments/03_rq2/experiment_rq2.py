"""
RQ2 Experiments: Hallucination Detection
==========================================
Three experiments that test WHETHER we can automatically catch the mistakes.

Experiment 4 (SelfCheckGPT): Ask same question 5 times, check if answers agree
Experiment 5 (MiniCheck):    Break answer into claims, check each against transcript
Experiment 6 (AlignScore):   Score each response's faithfulness to context (0-1)

Experiments 5 and 6 run LOCAL models (no API cost, runs on your PC).

Input:  {results_dir}/01_rag_responses.json
Output: {results_dir}/05_rq2_selfcheck.json
        {results_dir}/06_rq2_minicheck.json
        {results_dir}/07_rq2_alignscore.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

# Redirect HuggingFace model cache to project hf_cache folder
_HF_CACHE = Path(__file__).parent.parent / "hf_cache"
_HF_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE))


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.split()) >= 3]


def split_claims(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 4]


# ============================================================
# EXPERIMENT 4: SelfCheckGPT
# Paper: Manakul et al., EMNLP 2023
# ============================================================

def run_experiment_4(responses: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 4: SelfCheckGPT — Self-Consistency Detection.

    How it works:
    1. We already have the primary response + 5 sampled responses (from rag_pipeline.py)
    2. Split primary response into sentences
    3. For each sentence, measure how similar it is to the sampled responses
    4. Low similarity = the AI is making stuff up (hallucination)
    """
    print("=" * 60)
    print("EXPERIMENT 4: SelfCheckGPT (Self-Consistency)")
    print("Paper: Manakul et al., EMNLP 2023")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "05_rq2_selfcheck.json"

    try:
        from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
        selfcheck = SelfCheckBERTScore()
        print("SelfCheckGPT loaded (BERTScore variant)")
        USE_LIB = True
        HAS_BERTSCORE = False
    except Exception as e:
        if not isinstance(e, ImportError):
            print(f"selfcheckgpt failed ({type(e).__name__}). Trying bert_score fallback...")
        else:
            print("selfcheckgpt not installed. Trying bert_score fallback...")
        USE_LIB = False
        try:
            from bert_score import score as bert_score_fn
            print("Using bert_score directly")
            HAS_BERTSCORE = True
        except ImportError:
            print("bert_score not installed either. Using simple text overlap fallback.")
            HAS_BERTSCORE = False

    valid = [r for r in responses if r.get("rag_response") and r.get("sampled_responses")]
    print(f"\nProcessing {len(valid)} responses with sampled responses...")

    results = []
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}

    for i, r in enumerate(valid):
        key = (r["interview_id"], r["query_type"])
        if key in done_keys:
            continue

        sentences = split_sentences(r["rag_response"])
        if not sentences:
            continue

        print(f"\n[{i+1}/{len(valid)}] {r['query_type']}: {len(sentences)} sentences...")

        try:
            if USE_LIB:
                scores = selfcheck.predict(
                    sentences=sentences,
                    sampled_passages=r["sampled_responses"],
                )
                scores = [float(s) for s in scores]
            elif HAS_BERTSCORE:
                scores = []
                refs_all = r["sampled_responses"]
                for sent in sentences:
                    cands = [sent] * len(refs_all)
                    P, R, F1 = bert_score_fn(
                        cands, refs_all,
                        model_type="distilbert-base-uncased",
                        lang="en", verbose=False,
                        rescale_with_baseline=False,
                    )
                    avg_f1 = float(F1.mean())
                    scores.append(1.0 - avg_f1)
            else:
                scores = []
                for sent in sentences:
                    sent_words = set(sent.lower().split())
                    overlaps = []
                    for sample in r["sampled_responses"]:
                        sample_words = set(sample.lower().split())
                        if sent_words:
                            overlap = len(sent_words & sample_words) / len(sent_words)
                            overlaps.append(overlap)
                    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
                    scores.append(1.0 - avg_overlap)

            avg_score = sum(scores) / len(scores) if scores else 0
            result = {
                "interview_id": r["interview_id"],
                "language": r["language"],
                "query": r["query"],
                "query_type": r["query_type"],
                "rag_model": r.get("rag_model", "unknown"),
                "rag_response": r["rag_response"],
                "method": "selfcheckgpt_bertscore",
                "sentence_scores": scores,
                "avg_hallucination_score": round(avg_score, 4),
                "is_hallucinated": avg_score > 0.5,
            }
            results.append(result)
            done_keys.add(key)

            status = "HALLUCINATED" if avg_score > 0.5 else "OK"
            print(f"  Score: {avg_score:.3f} -> {status}")

        except Exception as e:
            print(f"  ERROR: {e}")

        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} SelfCheckGPT results to {output_path}")
    return results


# ============================================================
# EXPERIMENT 5: MiniCheck
# Paper: Tang, Laban & Durrett, EMNLP 2024
# ============================================================

def run_experiment_5(responses: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 5: MiniCheck — NLI-Based Claim Verification.

    How it works:
    1. Break the AI response into individual claims
    2. For each claim, check: is this claim supported by the transcript chunks?
    3. Unsupported claims = hallucinations
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: MiniCheck (Claim Verification)")
    print("Paper: Tang, Laban & Durrett, EMNLP 2024")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    output_path = out_dir / "06_rq2_minicheck.json"

    USE_MINICHECK = False
    USE_NLI = False

    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _mc_model_name = "lytang/MiniCheck-Flan-T5-Large"
        print(f"Loading MiniCheck model: {_mc_model_name}")
        mc_tokenizer = T5Tokenizer.from_pretrained(_mc_model_name, cache_dir=str(_HF_CACHE))
        mc_model = T5ForConditionalGeneration.from_pretrained(_mc_model_name, cache_dir=str(_HF_CACHE))
        mc_model.eval()
        USE_MINICHECK = True
        print("MiniCheck-Flan-T5-Large loaded (real paper model, CPU)")
    except Exception as e:
        print(f"MiniCheck model unavailable ({type(e).__name__}). Trying NLI fallback...")
        try:
            from transformers import pipeline
            nli_pipe = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,
                model_kwargs={"cache_dir": str(_HF_CACHE)},
            )
            USE_NLI = True
            print("Using NLI fallback: cross-encoder/nli-deberta-v3-small (CPU)")
        except Exception as e2:
            print(f"ERROR loading NLI model ({type(e2).__name__}): {e2}")
            print("Skipping MiniCheck experiment.")
            return []

    valid = [r for r in responses if r.get("rag_response")]
    print(f"\nProcessing {len(valid)} responses...")

    results = []
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}

    for i, r in enumerate(valid):
        key = (r["interview_id"], r["query_type"])
        if key in done_keys:
            continue

        claims = split_claims(r["rag_response"])
        if not claims:
            continue

        print(f"\n[{i+1}/{len(valid)}] Checking {len(claims)} claims...")
        claim_results = []

        for j, claim in enumerate(claims):
            try:
                if USE_MINICHECK:
                    prompt = f"premise: {r['context'][:2000]} hypothesis: {claim}"
                    inputs = mc_tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=512
                    )
                    with __import__("torch").no_grad():
                        outputs = mc_model.generate(**inputs, max_new_tokens=5)
                    pred = mc_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    is_supported = pred == "1"
                    confidence = 1.0 if is_supported else 0.0
                elif USE_NLI:
                    input_text = f"{r['context'][:1500]} [SEP] {claim}"
                    result_nli = nli_pipe(input_text, truncation=True)
                    label = result_nli[0]["label"].lower()
                    is_supported = "entail" in label
                    confidence = result_nli[0]["score"]

                claim_results.append({
                    "claim": claim,
                    "is_supported": is_supported,
                    "confidence": round(confidence, 3),
                })
                status = "SUPPORTED" if is_supported else "NOT SUPPORTED"
                print(f"  Claim {j+1}: {status} ({confidence:.2f}) — {claim[:50]}...")

            except Exception as e:
                print(f"  Claim {j+1} ERROR: {e}")
                claim_results.append({"claim": claim, "is_supported": None, "error": str(e)})

        supported = sum(1 for c in claim_results if c.get("is_supported") is True)
        unsupported = sum(1 for c in claim_results if c.get("is_supported") is False)

        results.append({
            "interview_id": r["interview_id"],
            "language": r["language"],
            "query": r["query"],
            "query_type": r["query_type"],
            "rag_model": r.get("rag_model", "unknown"),
            "rag_response": r["rag_response"],
            "method": "minicheck_flan_t5" if USE_MINICHECK else "minicheck_nli_fallback",
            "total_claims": len(claim_results),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "hallucination_ratio": round(unsupported / len(claim_results), 3) if claim_results else 0,
            "is_hallucinated": unsupported > 0,
            "claim_details": claim_results,
        })
        done_keys.add(key)

        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} MiniCheck results to {output_path}")
    return results


# ============================================================
# EXPERIMENT 6: AlignScore
# Paper: Zha et al., ACL 2023
# ============================================================

def run_experiment_6(responses: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 6: AlignScore — NLI-Based Faithfulness Scoring.

    Scores each (context, response) pair from 0 to 1.
    Score near 0 = hallucinated, near 1 = faithful.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: AlignScore (Faithfulness Scoring)")
    print("Paper: Zha et al., ACL 2023")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    output_path = out_dir / "07_rq2_alignscore.json"

    USE_ALIGNSCORE = False
    USE_NLI = False

    try:
        from alignscore import AlignScore as _AlignScore
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _device = "cpu"
        scorer = _AlignScore(
            model="roberta-large",
            batch_size=8,
            device=_device,
            evaluation_mode="nli_sp",
        )
        USE_ALIGNSCORE = True
        print(f"AlignScore loaded (roberta-large, device={_device})")
    except ImportError:
        print("alignscore not installed. Falling back to DeBERTa NLI scoring...")
        try:
            from transformers import pipeline as _pipeline
            nli_pipe = _pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,
                model_kwargs={"cache_dir": str(_HF_CACHE)},
            )
            USE_NLI = True
            print("NLI fallback loaded: cross-encoder/nli-deberta-v3-small")
        except Exception as e2:
            print(f"ERROR loading NLI fallback ({type(e2).__name__}): {e2}")
            print("Skipping AlignScore experiment.")
            return []
    except Exception as e:
        print(f"ERROR loading AlignScore ({type(e).__name__}): {e}")
        return []

    valid = [r for r in responses if r.get("rag_response") and r.get("context")]
    print(f"\nProcessing {len(valid)} responses...")

    results = []
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}

    for i, r in enumerate(valid):
        key = (r["interview_id"], r["query_type"])
        if key in done_keys:
            continue

        print(f"\n[{i+1}/{len(valid)}] {r['query_type']}: scoring faithfulness...")

        try:
            start = time.time()

            if USE_ALIGNSCORE:
                score = scorer.score(
                    contexts=[r["context"][:3000]],
                    claims=[r["rag_response"]],
                )[0]
                latency = time.time() - start
                method = "alignscore_roberta_large"
            elif USE_NLI:
                sentences = split_sentences(r["rag_response"])
                if not sentences:
                    sentences = [r["rag_response"]]
                context_text = r["context"][:1500]
                sentence_scores = []
                for sent in sentences:
                    input_text = f"{context_text} [SEP] {sent}"
                    result_nli = nli_pipe(input_text, truncation=True)
                    label = result_nli[0]["label"].lower()
                    raw_score = result_nli[0]["score"]
                    if "entail" in label:
                        sentence_scores.append(raw_score)
                    elif "contradict" in label:
                        sentence_scores.append(1.0 - raw_score)
                    else:
                        sentence_scores.append(0.5)
                score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.5
                latency = time.time() - start
                method = "alignscore_nli_fallback"

            is_hallucinated = score < 0.5
            results.append({
                "interview_id": r["interview_id"],
                "language": r["language"],
                "query": r["query"],
                "query_type": r["query_type"],
                "rag_model": r.get("rag_model", "unknown"),
                "rag_response": r["rag_response"],
                "method": method,
                "align_score": round(score, 4),
                "is_hallucinated": is_hallucinated,
                "latency": round(latency, 3),
            })
            done_keys.add(key)

            status = "HALLUCINATED" if is_hallucinated else "FAITHFUL"
            print(f"  AlignScore: {score:.3f} -> {status} [{latency:.2f}s]")

        except Exception as e:
            print(f"  ERROR: {e}")

        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} AlignScore results to {output_path}")
    return results


# ============================================================
# EXPERIMENT 7: RAGAS Faithfulness
# Paper: Es et al., EACL 2024
# ============================================================

def run_experiment_7(responses: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 7: RAGAS Faithfulness — Statement-Level Verification.

    How it works:
    1. RAGAS decomposes the response into atomic statements (LLM call)
    2. For each statement: checks if it can be inferred from context (LLM call)
    3. faithfulness_score = supported_statements / total_statements

    Evaluator: GPT-4o-mini (same OpenAI key used for RQ1 judge).
    No local model download required.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: RAGAS Faithfulness")
    print("Paper: Es et al., EACL 2024")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "08_rq2_ragas.json"

    # Load RAGAS and configure gpt-4o-mini as the LLM judge
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import faithfulness as ragas_faithfulness
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI
            ragas_faithfulness.llm = LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )
            print("RAGAS configured: gpt-4o-mini as LLM judge")
        except Exception as e:
            print(f"Could not set gpt-4o-mini ({e}); using RAGAS default LLM")
    except ImportError:
        print("ragas not installed. Run: pip install ragas>=0.2.0")
        return []

    valid = [r for r in responses if r.get("rag_response") and r.get("context")]
    print(f"\nProcessing {len(valid)} responses...")

    results = []
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}
        print(f"Resuming: {len(results)} already done")

    for i, r in enumerate(valid):
        key = (r["interview_id"], r["query_type"])
        if key in done_keys:
            continue

        print(f"\n[{i+1}/{len(valid)}] {r['query_type']}: RAGAS faithfulness scoring...")

        try:
            start = time.time()
            sample = SingleTurnSample(
                user_input=r["query"],
                response=r["rag_response"],
                retrieved_contexts=[r["context"][:3000]],
            )
            dataset = EvaluationDataset(samples=[sample])
            result_ragas = evaluate(dataset, metrics=[ragas_faithfulness])
            score_val = result_ragas["faithfulness"]
            score = float(score_val[0]) if isinstance(score_val, list) else float(score_val)
            latency = time.time() - start

            is_hallucinated = score < 0.5
            results.append({
                "interview_id": r["interview_id"],
                "language": r["language"],
                "query": r["query"],
                "query_type": r["query_type"],
                "rag_model": r.get("rag_model", "unknown"),
                "rag_response": r["rag_response"],
                "method": "ragas_faithfulness_eacl2024",
                "faithfulness_score": round(score, 4),
                "score": round(score, 4),   # alias for compare_models.py
                "is_hallucinated": is_hallucinated,
                "latency": round(latency, 3),
            })
            done_keys.add(key)

            status = "HALLUCINATED" if is_hallucinated else "FAITHFUL"
            print(f"  RAGAS faithfulness: {score:.3f} -> {status} [{latency:.1f}s]")

        except Exception as e:
            print(f"  ERROR: {e}")

        if (i + 1) % 5 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} RAGAS results to {output_path}")
    return results


# ============================================================
# Run all RQ2 experiments
# ============================================================

def run_all_rq2(results_dir: str = "results/gpt-4o-mini"):
    """Run all 3 RQ2 experiments for the given model's results dir."""
    input_path = Path(results_dir) / "01_rag_responses.json"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate step first.")
        return

    with open(input_path) as f:
        responses = json.load(f)

    model_name = responses[0].get("rag_model", "unknown") if responses else "unknown"
    print(f"\nLoaded {len(responses)} RAG responses (model: {model_name})")

    run_experiment_4(responses, results_dir=results_dir)
    run_experiment_5(responses, results_dir=results_dir)
    run_experiment_6(responses, results_dir=results_dir)
    run_experiment_7(responses, results_dir=results_dir)

    print("\n" + "=" * 60)
    print(f"ALL RQ2 EXPERIMENTS COMPLETE — {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/gpt-4o-mini")
    args = parser.parse_args()
    run_all_rq2(results_dir=args.results_dir)
