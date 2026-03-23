"""
Evaluate 4 RQ2 detectors with Precision, Recall, F1 vs manual labels.
================================================================
PURPOSE:
  Uses your 30 manual labels as ground truth to compute P/R/F1 for each
  of the 4 automated detectors (SelfCheckGPT, MiniCheck, AlignScore, RAGAS).
  This turns your RQ2 from "detector scores" to "which detector actually works."

HOW TO RUN:
  python D:\\RAG_THESIS\\validation\\detector_eval.py

PREREQUISITES:
  pip install scikit-learn
  Complete manual_validation_30.csv (YOUR_label column filled in)
  Run thesis pipeline so RQ2 JSON files exist in results/gpt-4o-mini/

OUTPUT:
  - Console: Precision/Recall/F1 table for all 4 detectors
  - D:\\RAG_THESIS\\output\\detector_f1_table.json

USE IN THESIS (Table 5.2):
  "Table 5.2 compares four automated hallucination detectors on the manually
  validated subset (N=30). [Best detector] achieves the highest F1 (X),
  suggesting NLI-based claim verification is most effective for interview RAG."
"""

import csv
import json
import sys
from pathlib import Path

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
except ImportError:
    print("ERROR: pip install scikit-learn")
    sys.exit(1)

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_BASE = Path(r"D:\thesis_hallucination\results\gpt-4o-mini")
OUTPUT_DIR   = Path(r"D:\RAG_THESIS\output")
INPUT_CSV    = OUTPUT_DIR / "manual_validation_30.csv"
OUTPUT_JSON  = OUTPUT_DIR / "detector_f1_table.json"

DETECTOR_FILES = {
    "SelfCheckGPT":  RESULTS_BASE / "05_rq2_selfcheck.json",
    "MiniCheck":     RESULTS_BASE / "06_rq2_minicheck.json",
    "AlignScore":    RESULTS_BASE / "07_rq2_alignscore.json",
    "RAGAS":         RESULTS_BASE / "08_rq2_ragas.json",
}

# Score fields for continuous AUC computation (higher = more hallucinated)
SCORE_FIELDS = {
    "SelfCheckGPT": "avg_hallucination_score",   # 0=faithful, 1=hallucinated
    "MiniCheck":    "hallucination_ratio",        # 0=supported, 1=unsupported
    "AlignScore":   None,                         # need to invert align_score
    "RAGAS":        None,                         # need to invert faithfulness_score
}


def load_detector_results(filepath: Path) -> dict:
    """Load detector JSON as dict keyed by (interview_id, query_type)."""
    if not filepath.exists():
        return {}
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    # Handle both list and wrapped-in-dict formats
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "evaluations" in data:
        items = data["evaluations"]
    else:
        items = data
    return {(item.get("interview_id"), item.get("query_type")): item for item in items}


def main():
    # ── load manual labels ───────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Run export_for_validation.py, fill in YOUR_label column, then retry.")
        sys.exit(1)

    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if r.get("YOUR_label", "").strip() in ("HALLUCINATED", "FAITHFUL")]
    if not labeled:
        print("ERROR: Fill in YOUR_label column first.")
        sys.exit(1)

    ground_truth = {
        (r["interview_id"], r["query_type"]): 1 if r["YOUR_label"].strip() == "HALLUCINATED" else 0
        for r in labeled
    }
    keys = list(ground_truth.keys())
    y_true = [ground_truth[k] for k in keys]
    n = len(keys)
    print(f"Evaluating {n} manually labeled samples against 4 detectors...\n")

    # ── evaluate each detector ───────────────────────────────────────────────
    results = {}
    all_results_table = []

    for name, filepath in DETECTOR_FILES.items():
        detector_data = load_detector_results(filepath)
        if not detector_data:
            print(f"  WARNING: {name} results not found at {filepath}")
            all_results_table.append({"detector": name, "status": "missing"})
            continue

        y_pred = []
        y_score = []
        matched = 0

        for key in keys:
            item = detector_data.get(key)
            if item is None:
                # Try to find by interview_id only (partial match)
                item = next(
                    (v for k, v in detector_data.items() if k[0] == key[0] and k[1] == key[1]),
                    None
                )
            if item:
                matched += 1
                y_pred.append(1 if item.get("is_hallucinated", False) else 0)
                # Continuous score for AUC
                if name == "SelfCheckGPT":
                    y_score.append(item.get("avg_hallucination_score", 0.5))
                elif name == "MiniCheck":
                    y_score.append(item.get("hallucination_ratio", 0.5))
                elif name == "AlignScore":
                    s = item.get("align_score", 0.5)
                    y_score.append(1.0 - s)     # invert: lower align = more hallucinated
                elif name == "RAGAS":
                    s = item.get("faithfulness_score", item.get("score", 0.5))
                    y_score.append(1.0 - s)     # invert: lower faithfulness = more hallucinated
            else:
                y_pred.append(0)    # default faithful if no match
                y_score.append(0.5)

        if matched == 0:
            print(f"  WARNING: {name} — no matching keys found. Check interview_id format.")
            continue

        y_true_matched = y_true[:len(y_pred)]
        p = precision_score(y_true_matched, y_pred, zero_division=0)
        r = recall_score(y_true_matched, y_pred, zero_division=0)
        f = f1_score(y_true_matched, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_true_matched, y_score)
        except Exception:
            auc = float("nan")

        results[name] = {"precision": p, "recall": r, "f1": f, "auc": auc, "matched": matched}
        all_results_table.append({
            "detector": name,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
            "auc": round(auc, 3) if auc == auc else "N/A",
            "samples_matched": matched,
        })

    # ── print table ──────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"{'Detector':<18} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'n':>5}")
    print("=" * 65)
    best_f1 = 0
    best_name = None
    for row in all_results_table:
        if "f1" not in row:
            print(f"  {row['detector']:<16}  [missing]")
            continue
        p, r, f, a = row["precision"], row["recall"], row["f1"], row["auc"]
        print(f"  {row['detector']:<16}  {p:>10.3f} {r:>8.3f} {f:>8.3f} {str(a):>8} {row['samples_matched']:>5}")
        if isinstance(f, float) and f > best_f1:
            best_f1 = f
            best_name = row["detector"]
    print("=" * 65)
    if best_name:
        print(f"\nBest detector: {best_name} (F1 = {best_f1:.3f})")

    # ── thesis table snippet ─────────────────────────────────────────────────
    print("\nUSE IN THESIS (Table 5.2 caption):")
    print(f'  "Automated detector evaluation on N={n} manually validated samples.')
    if best_name:
        print(f"  {best_name} achieves the highest F1 ({best_f1:.2f}), indicating that")
        print(f"  [NLI-based / self-consistency-based] verification is most")
        print(f'  effective for detecting hallucinations in interview RAG responses."')

    # ── save JSON ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"n_samples": n, "detectors": all_results_table, "best_detector": best_name}, f, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
