"""
Calculate Cohen's Kappa from manually annotated CSV.
================================================================
PURPOSE:
  Compares your manual hallucination labels against GPT-4o-mini auto-annotations.
  Computes Cohen's Kappa (κ) and raw agreement percentage for thesis validation.

HOW TO RUN:
  python D:\\RAG_THESIS\\validation\\calculate_kappa.py

PREREQUISITES:
  pip install scikit-learn
  Fill in YOUR_label column in: D:\\RAG_THESIS\\output\\manual_validation_30.csv
  (run export_for_validation.py first, then fill in Excel)

OUTPUT:
  - Console: full agreement statistics
  - D:\\RAG_THESIS\\output\\validation_results.json

INTERPRETING KAPPA:
  κ < 0.20  = Slight agreement       → your taxonomy may be unclear
  κ 0.21-0.40 = Fair agreement       → still acceptable if n is small
  κ 0.41-0.60 = Moderate agreement   → acceptable for thesis
  κ 0.61-0.80 = Substantial agreement → good, cite in thesis as "substantial"
  κ > 0.80  = Almost perfect          → excellent, cite as "near-perfect"

USE IN THESIS:
  "Manual verification of N=30 stratified responses confirms that the automated
  GPT-4o-mini annotator achieves X% raw agreement with human judgment (κ = Y),
  consistent with LLM-as-judge validation studies (Zheng et al., 2023)."
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter

try:
    from sklearn.metrics import cohen_kappa_score, classification_report
except ImportError:
    print("ERROR: run 'pip install scikit-learn' first")
    sys.exit(1)

# ── paths ────────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).parent.parent
OUTPUT_DIR  = _BASE / "output"
# Accept both the old 30-sample file and the new 200-sample file
INPUT_CSV   = OUTPUT_DIR / "manual_validation_200.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = OUTPUT_DIR / "manual_validation_30.csv"
OUTPUT_JSON = OUTPUT_DIR / "validation_results.json"


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Run export_for_validation.py first, then fill in YOUR_label in Excel.")
        sys.exit(1)

    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Filter to rows where human label is filled in
    labeled = [r for r in rows if r.get("YOUR_label", "").strip() in ("HALLUCINATED", "FAITHFUL")]
    total = len(labeled)

    if total == 0:
        print("ERROR: No rows have YOUR_label filled in yet.")
        print("Open the CSV in Excel and fill in the YOUR_label column.")
        sys.exit(1)

    human_labels = [r["YOUR_label"].strip() for r in labeled]
    auto_labels  = [r["gpt4_label"].strip() for r in labeled]

    # Filter out rows where gpt4_label is missing
    valid_pairs = [(h, a) for h, a in zip(human_labels, auto_labels) if a in ("HALLUCINATED", "FAITHFUL")]
    if not valid_pairs:
        print("ERROR: gpt4_label column is empty for all rows. Check your results files.")
        sys.exit(1)

    human_valid = [p[0] for p in valid_pairs]
    auto_valid  = [p[1] for p in valid_pairs]
    n = len(valid_pairs)

    # ── compute metrics ──────────────────────────────────────────────────────
    kappa = cohen_kappa_score(human_valid, auto_valid)
    raw_agreement = sum(h == a for h, a in zip(human_valid, auto_valid)) / n

    # Per-query-type breakdown
    by_type = {}
    for r, h, a in zip(labeled[:n], human_valid, auto_valid):
        qt = r.get("query_type", "unknown")
        by_type.setdefault(qt, {"agree": 0, "total": 0})
        by_type[qt]["total"] += 1
        if h == a:
            by_type[qt]["agree"] += 1

    # Confusion matrix counts
    tp = sum(1 for h, a in zip(human_valid, auto_valid) if h == "HALLUCINATED" and a == "HALLUCINATED")
    tn = sum(1 for h, a in zip(human_valid, auto_valid) if h == "FAITHFUL" and a == "FAITHFUL")
    fp = sum(1 for h, a in zip(human_valid, auto_valid) if h == "FAITHFUL" and a == "HALLUCINATED")
    fn = sum(1 for h, a in zip(human_valid, auto_valid) if h == "HALLUCINATED" and a == "FAITHFUL")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # ── kappa interpretation ─────────────────────────────────────────────────
    if kappa < 0.20:
        interp = "Slight agreement — taxonomy may need clarification"
    elif kappa < 0.41:
        interp = "Fair agreement — acceptable for small N"
    elif kappa < 0.61:
        interp = "Moderate agreement — acceptable for thesis"
    elif kappa < 0.81:
        interp = "Substantial agreement — cite as 'substantial' in thesis"
    else:
        interp = "Almost perfect agreement — cite as 'near-perfect' in thesis"

    # ── print results ────────────────────────────────────────────────────────
    print("=" * 60)
    print("MANUAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated   : {n}")
    print(f"  Raw agreement       : {raw_agreement:.1%}  ({sum(h==a for h,a in zip(human_valid, auto_valid))}/{n} correct)")
    print(f"  Cohen's Kappa (κ)   : {kappa:.3f}")
    print(f"  Interpretation      : {interp}")
    print()
    print("Confusion Matrix (GPT-4o-mini vs Human):")
    print(f"  True Positives  (both HALLUCINATED) : {tp}")
    print(f"  True Negatives  (both FAITHFUL)     : {tn}")
    print(f"  False Positives (GPT said HALL, you said FAITH) : {fp}")
    print(f"  False Negatives (GPT said FAITH, you said HALL) : {fn}")
    print()
    print("GPT-4o-mini Annotator Performance (vs human):")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print()
    print("Agreement by Query Type:")
    for qt, counts in sorted(by_type.items()):
        pct = counts["agree"] / counts["total"] if counts["total"] > 0 else 0
        bar = "█" * int(pct * 20)
        print(f"  {qt:<25} {pct:.0%}  {bar}")

    print()
    print("USE IN THESIS (copy-paste):")
    print(f'  "Manual verification of N={n} stratified responses (5 per query type)')
    print(f'  confirms that the automated GPT-4o-mini annotator achieves {raw_agreement:.0%}')
    print(f"  raw agreement with human judgment (κ = {kappa:.2f}), consistent with")
    print(f'  LLM-as-judge validation studies (Zheng et al., 2023)."')

    # ── save JSON ────────────────────────────────────────────────────────────
    results = {
        "n": n,
        "raw_agreement": round(raw_agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "interpretation": interp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "by_query_type": {qt: {"agreement": c["agree"]/c["total"], **c} for qt, c in by_type.items()},
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
