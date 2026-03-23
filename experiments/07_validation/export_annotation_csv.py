"""
Export 200 stratified samples for manual annotation.
======================================================
PURPOSE:
  Exports ~200 query-response pairs from RQ1 results for manual labeling.
  Stratified: ~33 per query type (6 types × 33 ≈ 200).
  Includes a short transcript excerpt so you can verify each response.

HOW TO RUN:
  cd /path/to/THESIS-main
  python RAG_THESIS/validation/export_annotation_csv.py

OUTPUT:
  RAG_THESIS/output/manual_validation_200.csv
  (Open in Excel or Google Sheets, fill in YOUR_label column)

YOUR_label VALUES:
  HALLUCINATED  — response contains false/unsupported claims
  FAITHFUL      — response accurately reflects the transcript
  REFUSAL       — response refuses to answer / says "not available" (even if info exists)
  PARTIAL       — partly correct, partly hallucinated (use sparingly)
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent.parent.parent          # THESIS-main/
RESULTS_DIR   = BASE / "results" / "gpt-4o-mini"
OUTPUT_DIR    = Path(__file__).parent.parent / "output"
OUTPUT_CSV    = OUTPUT_DIR / "manual_validation_200.csv"

ANNOTATIONS_FILE  = RESULTS_DIR / "02_rq1_annotations.json"
RESPONSES_FILE    = RESULTS_DIR / "01_rag_responses.json"

TARGET_TOTAL      = 200
QUERY_TYPES       = [
    "speaker_attribution",
    "participant_content",
    "factual_summary",
    "specific_content",
    "sentiment",
    "temporal",
]
TARGET_PER_TYPE   = TARGET_TOTAL // len(QUERY_TYPES)   # ~33

RANDOM_SEED = 42


def truncate(text: str, max_chars: int = 300) -> str:
    """Shorten text for the CSV so cells stay readable."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + "..." if len(text) > max_chars else text


def load_transcript_map(responses_file: Path) -> dict:
    """Build {(interview_id, query_type): transcript_excerpt} lookup."""
    if not responses_file.exists():
        return {}
    with open(responses_file, encoding="utf-8") as f:
        responses = json.load(f)
    mapping = {}
    for r in responses:
        key = (r.get("interview_id", ""), r.get("query_type", ""))
        transcript = r.get("transcript_text", r.get("context", ""))
        mapping[key] = truncate(transcript, 400)
    return mapping


def main():
    if not ANNOTATIONS_FILE.exists():
        print(f"ERROR: {ANNOTATIONS_FILE} not found.")
        print("Run the RQ1 experiment first: python src/run_all.py --steps rq1")
        raise SystemExit(1)

    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        annotations = json.load(f)

    transcript_map = load_transcript_map(RESPONSES_FILE)

    # ── stratified sampling ───────────────────────────────────────────────────
    by_type = defaultdict(list)
    for row in annotations:
        qt = row.get("query_type", "unknown")
        by_type[qt].append(row)

    random.seed(RANDOM_SEED)
    sampled = []
    for qt in QUERY_TYPES:
        pool = by_type.get(qt, [])
        if not pool:
            print(f"  WARNING: No samples for query_type='{qt}'")
            continue
        n = min(TARGET_PER_TYPE, len(pool))
        sampled.extend(random.sample(pool, n))
        print(f"  {qt:<25} → {n} samples (pool={len(pool)})")

    # Shuffle so all types are interleaved (harder to develop bias)
    random.shuffle(sampled)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "id",
        "query_type",
        "language",
        "query",
        "rag_response",
        "transcript_excerpt",
        "gpt4_label",
        "gpt4_faithfulness_score",
        "hallucination_types_detected",
        "YOUR_label",        # ← fill this in
        "notes",             # ← optional comments
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(sampled, 1):
            interview_id = row.get("interview_id", "")
            qt           = row.get("query_type", "")
            key          = (interview_id, qt)

            hall_types = row.get("hallucination_types", [])
            if isinstance(hall_types, list):
                hall_types_str = ", ".join(hall_types) if hall_types else "none"
            else:
                hall_types_str = str(hall_types)

            writer.writerow({
                "id":                        i,
                "query_type":                qt,
                "language":                  row.get("language", ""),
                "query":                     row.get("query", ""),
                "rag_response":              truncate(row.get("rag_response", ""), 500),
                "transcript_excerpt":        transcript_map.get(key, ""),
                "gpt4_label":                row.get("overall_label", ""),
                "gpt4_faithfulness_score":   row.get("faithfulness_score", ""),
                "hallucination_types_detected": hall_types_str,
                "YOUR_label":                "",   # to be filled manually
                "notes":                     "",
            })

    print(f"\nExported {len(sampled)} samples → {OUTPUT_CSV}")
    print()
    print("NEXT STEPS:")
    print("  1. Open manual_validation_200.csv in Excel or Google Sheets")
    print("  2. Add a Data Validation dropdown in the YOUR_label column:")
    print("       HALLUCINATED, FAITHFUL, REFUSAL, PARTIAL")
    print("  3. For each row:")
    print("       - Read the 'query' column (what was asked)")
    print("       - Read the 'rag_response' column (what the AI answered)")
    print("       - Read 'transcript_excerpt' (ground truth)")
    print("       - Ask: Does the response match the transcript? If yes → FAITHFUL")
    print("               Does it add false info?     → HALLUCINATED")
    print("               Does it refuse to answer?   → REFUSAL")
    print("  4. Work through ~30 rows per session (~30-45 min each)")
    print("  5. After filling in all rows, run:")
    print("       python RAG_THESIS/validation/calculate_kappa.py")
    print()
    print("TIP: If uncertain, write your reasoning in the 'notes' column.")
    print("TIP: Aim for 150+ labeled rows minimum (more = better Kappa reliability).")


if __name__ == "__main__":
    main()
