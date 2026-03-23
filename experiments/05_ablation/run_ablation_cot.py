"""
Ablation A5: CoT prompting vs production sentiment prompt.
================================================================
PURPOSE:
  Tests whether chain-of-thought (CoT) prompting reduces sentiment hallucination.
  Compares: SENTIMENT_SYSTEM_PROMPT (production baseline, ~76% hallucination)
       vs:  SENTIMENT_COT_PROMPT (CoT variant, expected ~40-50%)
  Both prompts are from convo_utils/sentiment_utils.py (extracted from company backend).

HOW TO RUN:
  python D:\\RAG_THESIS\\ablation\\run_ablation_cot.py

PREREQUISITES:
  pip install openai
  Set OPENAI_API_KEY in D:\\thesis_hallucination\\.env
  thesis_hallucination results must exist (01_rag_responses.json for gpt-4o-mini)

OUTPUT:
  - Console: baseline vs CoT hallucination rates for sentiment queries
  - D:\\RAG_THESIS\\output\\ablation_a5_cot_results.json

ESTIMATED COST: ~$0.10-0.30 (sentiment queries only, ~25-40 responses)
ESTIMATED TIME: 5-10 minutes

USE IN THESIS (Section 5.X — Ablation A5: Chain-of-Thought Prompting):
  "Replacing the production sentiment prompt with a CoT variant reduces sentiment
  query hallucination from 76.7% to X%, a reduction of Ypp. This confirms that
  explicit evidence citation (forced by CoT) mitigates sentiment inference errors,
  consistent with Wei et al. (2022)."
"""

import json
import os
import sys
import time
from pathlib import Path

# Load .env
_ENV = Path(r"D:\thesis_hallucination\.env")
if _ENV.exists():
    for line in _ENV.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Add convo_utils to path
sys.path.insert(0, str(Path(r"D:\thesis_hallucination\src")))

try:
    import openai
    client = openai.OpenAI()
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

try:
    from convo_utils.sentiment_utils import SENTIMENT_SYSTEM_PROMPT, SENTIMENT_COT_PROMPT
    HAS_CONVO = True
    print("Loaded SENTIMENT_SYSTEM_PROMPT + SENTIMENT_COT_PROMPT from convo_utils")
except ImportError:
    print("WARNING: convo_utils not available. Using built-in fallback prompts.")
    HAS_CONVO = False
    SENTIMENT_SYSTEM_PROMPT = (
        "You are a sentiment analyzer. Identify the overall sentiment of the participant "
        "in this interview excerpt. Answer the user's question based on the provided transcript."
    )
    SENTIMENT_COT_PROMPT = (
        "You are a sentiment analyzer. Before stating the sentiment, you MUST:\n"
        "1. Quote the EXACT words from the transcript that indicate the sentiment\n"
        "2. Explain what those words reveal about the participant's feelings\n"
        "3. Only then state your sentiment conclusion\n\n"
        "If you cannot find direct evidence in the transcript, say so explicitly."
    )

OUTPUT_DIR   = Path(r"D:\RAG_THESIS\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON  = OUTPUT_DIR / "ablation_a5_cot_results.json"
RESUME_FILE  = OUTPUT_DIR / "ablation_a5_partial.json"

RESULTS_DIR      = Path(r"D:\thesis_hallucination\results\gpt-4o-mini")
RESPONSES_FILE   = RESULTS_DIR / "01_rag_responses.json"
ANNOTATIONS_FILE = RESULTS_DIR / "02_rq1_annotations.json"

ANNOTATION_PROMPT = """\
You are an expert annotator for hallucination detection in interview-based RAG systems.

Compare the AI's response against the original interview transcript.

ORIGINAL TRANSCRIPT:
{transcript}

CONTEXT CHUNKS:
{context}

QUERY: {query}

AI RESPONSE:
{response}

Return JSON:
{{
  "overall_label": "HALLUCINATED" or "FAITHFUL",
  "faithfulness_score": 0.0 to 1.0,
  "hallucinations": [
    {{
      "type": "SENTIMENT_MISREPRESENTATION or other type",
      "span": "exact text from response",
      "severity": "LOW/MEDIUM/HIGH"
    }}
  ]
}}"""


def annotate(transcript, context, query, response):
    prompt = ANNOTATION_PROMPT.format(
        transcript=transcript[:3000], context=context[:2000],
        query=query, response=response
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=600,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def generate_sentiment_response(system_prompt, context, query):
    """Generate a response using the given system prompt."""
    full_system = f"{system_prompt}\n\nINTERVIEW TRANSCRIPT:\n{context}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": query},
        ],
        temperature=0.1, max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def main():
    print("=" * 60)
    print("ABLATION A5: CoT Prompting vs Production Sentiment Prompt")
    print("=" * 60)
    print(f"Baseline prompt source: {'convo_utils (production)' if HAS_CONVO else 'built-in fallback'}")

    # ── load data ────────────────────────────────────────────────────────────
    if not RESPONSES_FILE.exists():
        print(f"ERROR: {RESPONSES_FILE} not found. Run thesis pipeline first.")
        sys.exit(1)

    with open(RESPONSES_FILE, encoding="utf-8") as f:
        all_responses = json.load(f)

    # Filter to sentiment queries only
    sentiment_responses = [r for r in all_responses if "sentiment" in r.get("query_type", "").lower()]
    print(f"Sentiment queries found: {len(sentiment_responses)}")

    if not sentiment_responses:
        print("WARNING: No sentiment queries found. Check query_type field in your data.")
        print("Available query types:", set(r.get("query_type") for r in all_responses))
        sys.exit(0)

    # ── resume support ───────────────────────────────────────────────────────
    results = []
    done_keys = set()
    if RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}
        print(f"Resuming: {len(done_keys)} already done\n")

    # ── run ablation ─────────────────────────────────────────────────────────
    for i, r in enumerate(sentiment_responses):
        key = (r.get("interview_id"), r.get("query_type"))
        if key in done_keys:
            continue

        print(f"\n[{i+1}/{len(sentiment_responses)}] {r.get('interview_id')} | {r.get('query_type')}")

        transcript = r.get("transcript_text", "")[:4000]
        context    = r.get("context", "")[:2000]
        query      = r.get("query", "")
        original_response = r.get("rag_response", "")

        row = {
            "interview_id": r.get("interview_id"),
            "query_type":   r.get("query_type"),
            "query":        query,
            "language":     r.get("language", ""),
        }

        # ── CONDITION A: Annotate original response (baseline prompt) ──────
        try:
            baseline_ann = annotate(transcript, context, query, original_response)
            row["baseline_response"] = original_response
            row["baseline_label"]    = baseline_ann.get("overall_label", "UNKNOWN")
            row["baseline_faith"]    = baseline_ann.get("faithfulness_score", 0)
            row["baseline_n_issues"] = len(baseline_ann.get("hallucinations", []))
            print(f"  Baseline: {row['baseline_label']} | faith={row['baseline_faith']:.2f}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR annotating baseline: {e}")
            continue

        # ── CONDITION B: Generate + annotate CoT response ──────────────────
        try:
            cot_response = generate_sentiment_response(SENTIMENT_COT_PROMPT, context, query)
            time.sleep(0.5)

            cot_ann = annotate(transcript, context, query, cot_response)
            row["cot_response"] = cot_response
            row["cot_label"]    = cot_ann.get("overall_label", "UNKNOWN")
            row["cot_faith"]    = cot_ann.get("faithfulness_score", 0)
            row["cot_n_issues"] = len(cot_ann.get("hallucinations", []))
            row["improved"]     = row["cot_label"] == "FAITHFUL" and row["baseline_label"] == "HALLUCINATED"

            print(f"  CoT:      {row['cot_label']} | faith={row['cot_faith']:.2f}  {'✓ IMPROVED' if row['improved'] else ''}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR generating/annotating CoT: {e}")
            continue

        results.append(row)
        done_keys.add(key)

        if len(results) % 5 == 0:
            with open(RESUME_FILE, "w") as f:
                json.dump(results, f, indent=2)

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results. Check your data.")
        sys.exit(0)

    baseline_hall = sum(1 for r in results if r.get("baseline_label") == "HALLUCINATED") / n
    cot_hall      = sum(1 for r in results if r.get("cot_label") == "HALLUCINATED") / n
    improvement   = (baseline_hall - cot_hall) * 100

    avg_baseline_faith = sum(r.get("baseline_faith", 0) for r in results) / n
    avg_cot_faith      = sum(r.get("cot_faith", 0) for r in results) / n

    print("\n" + "=" * 60)
    print("ABLATION A5 RESULTS")
    print("=" * 60)
    print(f"  Sentiment responses evaluated: {n}")
    print(f"  Baseline (production prompt) hallucination rate: {baseline_hall:.1%}")
    print(f"  CoT prompt hallucination rate:                   {cot_hall:.1%}")
    print(f"  Reduction:                                       {improvement:+.1f}pp")
    print(f"  Baseline avg faithfulness: {avg_baseline_faith:.3f}")
    print(f"  CoT avg faithfulness:      {avg_cot_faith:.3f} (+{avg_cot_faith-avg_baseline_faith:.3f})")

    print(f"\nUSE IN THESIS:")
    print(f'  "Replacing the production sentiment prompt (SENTIMENT_SYSTEM_PROMPT)')
    print(f"  with a chain-of-thought variant (SENTIMENT_COT_PROMPT) reduces")
    print(f"  sentiment query hallucination from {baseline_hall:.1%} to {cot_hall:.1%}")
    print(f"  ({improvement:+.1f}pp), confirming that explicit evidence citation")
    print(f"  mitigates sentiment inference errors (Wei et al., 2022).")
    print(f"  Average faithfulness increases by {avg_cot_faith-avg_baseline_faith:.3f} points.")
    print(f'  This corresponds to Ablation A5 in our ablation study (Table 6.X)."')

    # ── save ─────────────────────────────────────────────────────────────────
    summary = {
        "n": n,
        "baseline_hallucination_rate": round(baseline_hall, 4),
        "cot_hallucination_rate": round(cot_hall, 4),
        "improvement_pp": round(improvement, 2),
        "avg_faithfulness_baseline": round(avg_baseline_faith, 4),
        "avg_faithfulness_cot": round(avg_cot_faith, 4),
        "faithfulness_gain": round(avg_cot_faith - avg_baseline_faith, 4),
        "prompt_source": "convo_utils/sentiment_utils.py" if HAS_CONVO else "built-in fallback",
        "individual_results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    if RESUME_FILE.exists():
        RESUME_FILE.unlink()

    print(f"\nResults saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
