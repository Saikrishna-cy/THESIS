"""
Build Unified Results File
===========================
Merges RQ1 annotations + RQ2 detector scores for all models into a single
JSONL file that RAG_THESIS advanced modules can consume.

Output: data/unified_results.jsonl

Each line:
{
  "query_id":            "<interview_id>__<query_type>",
  "interview_id":        "...",
  "query_type":          "...",
  "language":            "en|nl",
  "model":               "gpt-4o-mini|qwen|mistral",
  "query":               "...",
  "rag_response":        "...",

  # RQ1 — taxonomy labels
  "rq1_label":           "HALLUCINATED|FAITHFUL",
  "rq1_faithfulness":    0.0–1.0,
  "rq1_types":           ["REFUSAL_HALLUCINATION", ...],

  # RQ2 — detector scores (null if not available)
  "selfcheck_score":     0.0–1.0 or null,
  "selfcheck_label":     true|false or null,
  "minicheck_score":     0.0–1.0 or null,
  "minicheck_label":     true|false or null,
  "alignscore":          0.0–1.0 or null,
  "ragas_faithfulness":  0.0–1.0 or null,
}

Usage:
  python src/build_unified_results.py
  python src/build_unified_results.py --results-base results --output data/unified_results.jsonl
"""

import argparse
import json
from pathlib import Path

BASE        = Path(__file__).parent.parent
DEFAULT_RES = BASE / "results"
DEFAULT_OUT = BASE / "data" / "unified_results.jsonl"

MODELS = ["gpt-4o-mini", "qwen", "mistral"]

FILE_MAP = {
    "rq1":       "02_rq1_annotations.json",
    "selfcheck":  "05_rq2_selfcheck.json",
    "minicheck":  "06_rq2_minicheck.json",
    "alignscore": "07_rq2_alignscore.json",
    "ragas":      "08_rq2_ragas.json",
}


def _key(row: dict) -> str:
    return f"{row.get('interview_id', '')}___{row.get('query_type', '')}"


def load_model(results_base: Path, model: str) -> dict:
    """Load all available result files for one model. Returns dict keyed by query_id."""
    safe = model.replace("/", "-").replace(".", "-")
    mdir = results_base / safe

    # RQ1 is required
    rq1_path = mdir / FILE_MAP["rq1"]
    if not rq1_path.exists():
        print(f"  [{model}] No RQ1 file — skipping")
        return {}

    with open(rq1_path, encoding="utf-8") as f:
        rq1_rows = json.load(f)

    # Build base records from RQ1
    records = {}
    for row in rq1_rows:
        qid = _key(row)
        hall_types = row.get("hallucination_types", [])
        records[qid] = {
            "query_id":          qid,
            "interview_id":      row.get("interview_id", ""),
            "query_type":        row.get("query_type", ""),
            "language":          row.get("language", ""),
            "model":             model,
            "query":             row.get("query", ""),
            "rag_response":      row.get("rag_response", ""),
            "rq1_label":         row.get("overall_label", ""),
            "rq1_faithfulness":  row.get("faithfulness_score"),
            "rq1_types":         hall_types if isinstance(hall_types, list) else [],
            "selfcheck_score":   None,
            "selfcheck_label":   None,
            "minicheck_score":   None,
            "minicheck_label":   None,
            "alignscore":        None,
            "ragas_faithfulness": None,
        }

    # Merge RQ2 — selfcheck
    sc_path = mdir / FILE_MAP["selfcheck"]
    if sc_path.exists():
        with open(sc_path, encoding="utf-8") as f:
            for row in json.load(f):
                qid = _key(row)
                if qid in records:
                    records[qid]["selfcheck_score"] = row.get("avg_hallucination_score")
                    records[qid]["selfcheck_label"] = row.get("is_hallucinated")
        print(f"  [{model}] SelfCheck merged")
    else:
        print(f"  [{model}] SelfCheck not found (pending)")

    # Merge RQ2 — minicheck
    mc_path = mdir / FILE_MAP["minicheck"]
    if mc_path.exists():
        with open(mc_path, encoding="utf-8") as f:
            for row in json.load(f):
                qid = _key(row)
                if qid in records:
                    records[qid]["minicheck_score"] = row.get("hallucination_ratio")
                    records[qid]["minicheck_label"] = row.get("is_hallucinated")
        print(f"  [{model}] MiniCheck merged")
    else:
        print(f"  [{model}] MiniCheck not found (pending)")

    # Merge RQ2 — alignscore
    as_path = mdir / FILE_MAP["alignscore"]
    if as_path.exists():
        with open(as_path, encoding="utf-8") as f:
            for row in json.load(f):
                qid = _key(row)
                if qid in records:
                    records[qid]["alignscore"] = row.get("alignscore")
        print(f"  [{model}] AlignScore merged")
    else:
        print(f"  [{model}] AlignScore not found (pending)")

    # Merge RQ2 — ragas
    rg_path = mdir / FILE_MAP["ragas"]
    if rg_path.exists():
        with open(rg_path, encoding="utf-8") as f:
            for row in json.load(f):
                qid = _key(row)
                if qid in records:
                    records[qid]["ragas_faithfulness"] = row.get("faithfulness")
        print(f"  [{model}] RAGAS merged")
    else:
        print(f"  [{model}] RAGAS not found (pending)")

    return records


def build(results_base: Path, output_path: Path):
    print(f"\nBuilding unified results from: {results_base}")

    all_records = []
    for model in MODELS:
        print(f"\nModel: {model}")
        records = load_model(results_base, model)
        all_records.extend(records.values())
        print(f"  → {len(records)} records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(all_records)} records → {output_path}")

    # Coverage summary
    has_selfcheck  = sum(1 for r in all_records if r["selfcheck_score"] is not None)
    has_minicheck  = sum(1 for r in all_records if r["minicheck_score"] is not None)
    has_alignscore = sum(1 for r in all_records if r["alignscore"] is not None)
    has_ragas      = sum(1 for r in all_records if r["ragas_faithfulness"] is not None)

    print(f"\nCoverage:")
    print(f"  RQ1 labels    : {len(all_records)}/{len(all_records)} (100%)")
    print(f"  SelfCheck     : {has_selfcheck}/{len(all_records)} ({100*has_selfcheck//max(len(all_records),1)}%)")
    print(f"  MiniCheck     : {has_minicheck}/{len(all_records)} ({100*has_minicheck//max(len(all_records),1)}%)")
    print(f"  AlignScore    : {has_alignscore}/{len(all_records)} ({100*has_alignscore//max(len(all_records),1)}%)")
    print(f"  RAGAS         : {has_ragas}/{len(all_records)} ({100*has_ragas//max(len(all_records),1)}%)")


def main():
    parser = argparse.ArgumentParser(description="Build unified_results.jsonl")
    parser.add_argument("--results-base", default=str(DEFAULT_RES))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    build(Path(args.results_base), Path(args.output))


if __name__ == "__main__":
    main()
