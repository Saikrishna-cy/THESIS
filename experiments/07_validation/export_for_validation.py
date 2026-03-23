"""
Export 30 stratified responses to CSV for manual annotation.
================================================================
PURPOSE:
  Reads RAG responses and GPT-4o-mini auto-annotations from thesis_hallucination results.
  Exports a CSV with 30 rows (5 per query type) for YOU to fill in manually in Excel.

HOW TO RUN:
  python D:\\RAG_THESIS\\validation\\export_for_validation.py

PREREQUISITES:
  - thesis_hallucination results must exist:
      D:\\thesis_hallucination\\results\\gpt-4o-mini\\01_rag_responses.json
      D:\\thesis_hallucination\\results\\gpt-4o-mini\\02_rq1_annotations.json

OUTPUT:
  D:\\RAG_THESIS\\output\\manual_validation_30.csv

WHAT TO DO AFTER:
  1. Open manual_validation_30.csv in Excel
  2. For each row: read transcript_excerpt and ai_response
  3. Fill in YOUR_label column: HALLUCINATED or FAITHFUL
  4. Fill in YOUR_types column (comma-separated from these 7 types):
       EVIDENT_CONFLICT, SUBTLE_CONFLICT, BASELESS_INFO,
       SPEAKER_MISATTRIBUTION, TEMPORAL_CONFUSION,
       SENTIMENT_MISREPRESENTATION, REFUSAL_HALLUCINATION
  5. Add any notes in the notes column
  6. Save the CSV
  7. Run calculate_kappa.py to compute Cohen's Kappa
"""

import csv
import json
import random
import sys
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
THESIS_BASE  = Path(r"D:\thesis_hallucination")
RESULTS_BASE = THESIS_BASE / "results" / "gpt-4o-mini"
OUTPUT_DIR   = Path(r"D:\RAG_THESIS\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_FILE   = RESULTS_BASE / "01_rag_responses.json"
ANNOTATIONS_FILE = RESULTS_BASE / "02_rq1_annotations.json"
OUTPUT_CSV       = OUTPUT_DIR / "manual_validation_30.csv"

# ── query types to stratify over ────────────────────────────────────────────
QUERY_TYPES = [
    "speaker_attribution",
    "sentiment",
    "timeline",
    "satisfaction",
    "challenge",
    "improvement",
]

SAMPLES_PER_TYPE = 5   # 5 × 6 types = 30 total


def main():
    # ── load data ────────────────────────────────────────────────────────────
    if not RESPONSES_FILE.exists():
        print(f"ERROR: {RESPONSES_FILE} not found. Run thesis pipeline first.")
        sys.exit(1)

    with open(RESPONSES_FILE, encoding="utf-8") as f:
        responses = json.load(f)

    annotations = []
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
            annotations = json.load(f)
    else:
        print(f"WARNING: {ANNOTATIONS_FILE} not found. GPT-4o-mini labels will be empty.")

    # Index annotations
    ann_idx = {
        (a.get("interview_id"), a.get("query_type")): a
        for a in annotations
    }

    # ── stratified sample ────────────────────────────────────────────────────
    by_type = {}
    for r in responses:
        qt = r.get("query_type", "unknown")
        by_type.setdefault(qt, []).append(r)

    print(f"Loaded {len(responses)} responses across {len(by_type)} query types:")
    for qt, items in sorted(by_type.items()):
        print(f"  {qt}: {len(items)} responses")

    random.seed(42)  # reproducible
    selected = []
    for qt in QUERY_TYPES:
        pool = by_type.get(qt, [])
        if not pool:
            print(f"  WARNING: no responses for query type '{qt}'")
            continue
        sample = random.sample(pool, min(SAMPLES_PER_TYPE, len(pool)))
        for r in sample:
            ann = ann_idx.get((r.get("interview_id"), r.get("query_type")), {})
            selected.append({
                "interview_id":      r.get("interview_id", ""),
                "query_type":        r.get("query_type", ""),
                "language":          r.get("language", ""),
                "query":             r.get("query", ""),
                "transcript_excerpt": r.get("transcript_text", "")[:800],
                "context_given":     r.get("context", "")[:600],
                "ai_response":       r.get("rag_response", ""),
                "gpt4_label":        ann.get("overall_label", ""),
                "gpt4_types":        ", ".join(ann.get("hallucination_types", [])),
                "gpt4_faithfulness": ann.get("faithfulness_score", ""),
                "YOUR_label":        "",     # ← YOU FILL THIS IN
                "YOUR_types":        "",     # ← YOU FILL THIS IN
                "notes":             "",
            })

    print(f"\nExporting {len(selected)} responses to {OUTPUT_CSV}")

    # ── write CSV ────────────────────────────────────────────────────────────
    if not selected:
        print("ERROR: No responses selected. Check query type names match your data.")
        sys.exit(1)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    print(f"Done. Open {OUTPUT_CSV} in Excel and fill in YOUR_label + YOUR_types columns.")
    print("\nColumn guide:")
    print("  YOUR_label  : HALLUCINATED or FAITHFUL")
    print("  YOUR_types  : comma-separated hallucination types from the 7 types listed above")
    print("\nAfter filling in, run: python D:\\RAG_THESIS\\validation\\calculate_kappa.py")


if __name__ == "__main__":
    main()
