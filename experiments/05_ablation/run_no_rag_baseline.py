"""
Baseline experiment: No RAG context (direct LLM without transcript).
================================================================
PURPOSE:
  Establishes how much RAG actually helps by comparing:
    WITH RAG:    model sees retrieved transcript chunks → answers query
    WITHOUT RAG: model answers same query from world knowledge alone
  Expected: NO-RAG hallucination rate >> RAG hallucination rate.
  This shows RAG is essential and quantifies its benefit.

HOW TO RUN:
  python D:\\RAG_THESIS\\ablation\\run_no_rag_baseline.py

PREREQUISITES:
  pip install openai
  Set OPENAI_API_KEY in D:\\thesis_hallucination\\.env
  thesis_hallucination results must exist (01 and 02 JSON files)

OUTPUT:
  - Console: RAG vs No-RAG hallucination rates
  - D:\\RAG_THESIS\\output\\no_rag_baseline_results.json

ESTIMATED COST: ~$0.50-1.50 (150 responses × 2 API calls each)
ESTIMATED TIME: 15-30 minutes

USE IN THESIS (Section 4.X — Baseline: RAG vs No-RAG):
  "Without RAG context, GPT-4o-mini hallucinates in X% of responses.
  Adding interview transcripts via RAG reduces this to 56.7% (−Ypp reduction),
  demonstrating that grounding in retrieved context is essential but insufficient
  for eliminating hallucination in interview analysis tasks."
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

try:
    import openai
    client = openai.OpenAI()
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

OUTPUT_DIR   = Path(r"D:\RAG_THESIS\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON  = OUTPUT_DIR / "no_rag_baseline_results.json"
RESUME_FILE  = OUTPUT_DIR / "no_rag_partial.json"

RESULTS_DIR      = Path(r"D:\thesis_hallucination\results\gpt-4o-mini")
RESPONSES_FILE   = RESULTS_DIR / "01_rag_responses.json"

NO_RAG_SYSTEM = (
    "You are a research assistant helping analyze qualitative research data. "
    "Answer the user's question as best you can based on your general knowledge. "
    "Be specific and honest if you don't have enough information."
)

ANNOTATION_PROMPT = """\
You are an expert annotator for hallucination detection in interview-based RAG systems.

Compare the AI's response against the original interview transcript.
The AI did NOT have access to the transcript — judge whether its response
happens to be consistent with the transcript or contradicts/fabricates.

ORIGINAL TRANSCRIPT (ground truth):
{transcript}

QUERY: {query}

AI RESPONSE (no context was given to this AI):
{response}

Return JSON:
{{
  "overall_label": "HALLUCINATED" or "FAITHFUL",
  "faithfulness_score": 0.0 to 1.0,
  "note": "brief explanation"
}}"""


def generate_no_rag_response(query: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": NO_RAG_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.1, max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def annotate_no_rag(transcript: str, query: str, response: str) -> dict:
    prompt = ANNOTATION_PROMPT.format(
        transcript=transcript[:3000], query=query, response=response
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=400,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def main():
    print("=" * 60)
    print("BASELINE: No-RAG vs RAG Hallucination Comparison")
    print("=" * 60)

    if not RESPONSES_FILE.exists():
        print(f"ERROR: {RESPONSES_FILE} not found.")
        sys.exit(1)

    with open(RESPONSES_FILE, encoding="utf-8") as f:
        responses = json.load(f)

    valid = [r for r in responses if r.get("rag_response") and r.get("transcript_text")]
    print(f"Total responses to evaluate: {len(valid)}")

    # ── resume support ───────────────────────────────────────────────────────
    results = []
    done_keys = set()
    if RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}
        print(f"Resuming: {len(done_keys)} already done\n")

    # ── run experiment ───────────────────────────────────────────────────────
    for i, r in enumerate(valid):
        key = (r.get("interview_id"), r.get("query_type"))
        if key in done_keys:
            continue

        print(f"[{i+1}/{len(valid)}] {r.get('query_type')} | {r.get('interview_id')}")

        # Step 1: Generate response WITHOUT any context
        try:
            no_rag_response = generate_no_rag_response(r["query"])
            time.sleep(0.4)
        except Exception as e:
            print(f"  ERROR generating: {e}")
            continue

        # Step 2: Annotate no-RAG response against transcript
        try:
            ann = annotate_no_rag(r["transcript_text"], r["query"], no_rag_response)
            label = ann.get("overall_label", "UNKNOWN")
            faith = ann.get("faithfulness_score", 0)
            print(f"  No-RAG: {label} | faithfulness={faith:.2f}")
            time.sleep(0.4)
        except Exception as e:
            print(f"  ERROR annotating: {e}")
            continue

        results.append({
            "interview_id":       r.get("interview_id"),
            "query_type":         r.get("query_type"),
            "language":           r.get("language", ""),
            "query":              r.get("query", ""),
            "rag_response":       r.get("rag_response", ""),
            "no_rag_response":    no_rag_response,
            "no_rag_label":       label,
            "no_rag_faithfulness": faith,
            # RAG label from existing annotation (if available)
            "rag_label":          r.get("rag_label", ""),
        })
        done_keys.add(key)

        if len(results) % 10 == 0:
            with open(RESUME_FILE, "w") as f:
                json.dump(results, f, indent=2)

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results.")
        sys.exit(0)

    no_rag_hall_rate = sum(1 for r in results if r["no_rag_label"] == "HALLUCINATED") / n
    rag_hall_rate    = 0.567  # your existing result — update with actual

    avg_no_rag_faith = sum(r["no_rag_faithfulness"] for r in results) / n

    # By query type
    by_type = {}
    for r in results:
        qt = r["query_type"]
        by_type.setdefault(qt, {"hallucinated": 0, "total": 0})
        by_type[qt]["total"] += 1
        if r["no_rag_label"] == "HALLUCINATED":
            by_type[qt]["hallucinated"] += 1

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"  Responses evaluated: {n}")
    print(f"  No-RAG hallucination rate: {no_rag_hall_rate:.1%}")
    print(f"  RAG hallucination rate:    {rag_hall_rate:.1%}  (from existing results)")
    print(f"  RAG reduction:             {(no_rag_hall_rate - rag_hall_rate)*100:+.1f}pp")
    print(f"  No-RAG avg faithfulness:   {avg_no_rag_faith:.3f}")

    print("\nNo-RAG Hallucination Rate by Query Type:")
    for qt, counts in sorted(by_type.items(), key=lambda x: x[1]["hallucinated"]/x[1]["total"], reverse=True):
        rate = counts["hallucinated"] / counts["total"]
        bar = "█" * int(rate * 20)
        print(f"  {qt:<28} {rate:.1%}  {bar}")

    print(f"\nUSE IN THESIS:")
    print(f'  "Without RAG context, GPT-4o-mini hallucinates in {no_rag_hall_rate:.1%}')
    print(f"  of responses. Adding interview transcripts via RAG reduces this to")
    print(f"  {rag_hall_rate:.1%} (−{(no_rag_hall_rate-rag_hall_rate)*100:.1f}pp), demonstrating that grounding")
    print(f"  in retrieved context is necessary but insufficient for eliminating")
    print(f'  hallucination in interview analysis tasks (Table 4.X)."')

    # ── save ─────────────────────────────────────────────────────────────────
    output = {
        "n": n,
        "no_rag_hallucination_rate": round(no_rag_hall_rate, 4),
        "rag_hallucination_rate": rag_hall_rate,
        "rag_reduction_pp": round((no_rag_hall_rate - rag_hall_rate) * 100, 2),
        "avg_no_rag_faithfulness": round(avg_no_rag_faith, 4),
        "by_query_type": {qt: {"rate": c["hallucinated"]/c["total"], **c} for qt, c in by_type.items()},
        "individual_results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    if RESUME_FILE.exists():
        RESUME_FILE.unlink()

    print(f"\nResults saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
