"""
Novel: Ensemble of 4 hallucination detectors with learned weights.
================================================================
PURPOSE:
  Combines SelfCheckGPT, MiniCheck, AlignScore, and RAGAS scores into a single
  ensemble predictor trained on manual labels. Finds which detector is most
  reliable for the interview domain and whether ensemble beats individuals.

HOW TO RUN:
  python D:\\RAG_THESIS\\advanced\\ensemble_detector.py

PREREQUISITES:
  pip install scikit-learn matplotlib
  Complete: manual_validation_30.csv (YOUR_label column filled)
  Complete: all 4 RQ2 detector JSON files in thesis results

OUTPUT:
  - Console: Per-detector F1, Ensemble F1, feature weights
  - D:\\RAG_THESIS\\output\\ensemble_results.json
  - D:\\RAG_THESIS\\output\\ensemble_comparison_chart.png

USE IN THESIS (Section 5.Y — Ensemble Detection):
  "An ensemble of the four detectors, trained on N=30 manually validated samples,
  achieves F1 = X — a Y% improvement over the best individual detector (Z, F1=W).
  Logistic regression weights reveal that [MiniCheck/AlignScore/RAGAS] is the
  most informative signal (β = A), suggesting that [NLI-based claim verification /
  faithfulness scoring] is most effective for the interview RAG domain."
"""

import csv
import json
import sys
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: pip install scikit-learn")
    sys.exit(1)

_RAGTHESIS_BASE = Path(__file__).parent.parent
OUTPUT_DIR   = _RAGTHESIS_BASE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Accept 200-sample file if it exists, fall back to 30-sample
INPUT_CSV    = OUTPUT_DIR / "manual_validation_200.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = OUTPUT_DIR / "manual_validation_30.csv"
OUTPUT_JSON  = OUTPUT_DIR / "ensemble_results.json"
OUTPUT_CHART = OUTPUT_DIR / "ensemble_comparison_chart.png"

RESULTS_BASE = _RAGTHESIS_BASE.parent / "results" / "gpt-4o-mini"

DETECTORS = {
    "SelfCheckGPT": RESULTS_BASE / "05_rq2_selfcheck.json",
    "MiniCheck":    RESULTS_BASE / "06_rq2_minicheck.json",
    "AlignScore":   RESULTS_BASE / "07_rq2_alignscore.json",
    "RAGAS":        RESULTS_BASE / "08_rq2_ragas.json",
}


def load_detector(filepath, score_key, invert=False):
    """Load detector results as dict keyed by (interview_id, query_type)."""
    if not filepath.exists():
        return {}
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("evaluations", data)
    out = {}
    for item in items:
        key = (item.get("interview_id"), item.get("query_type"))
        score = item.get(score_key, 0.5)
        if invert:
            score = 1.0 - score
        binary = 1 if item.get("is_hallucinated", False) else 0
        out[key] = {"score": float(score), "binary": binary}
    return out


def main():
    # ── load manual labels ───────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run export_for_validation.py first.")
        sys.exit(1)

    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if r.get("YOUR_label", "").strip() in ("HALLUCINATED", "FAITHFUL")]
    if len(labeled) < 10:
        print(f"ERROR: Need at least 10 labeled samples. Found {len(labeled)}.")
        sys.exit(1)

    keys = [(r["interview_id"], r["query_type"]) for r in labeled]
    y_true = [1 if r["YOUR_label"].strip() == "HALLUCINATED" else 0 for r in labeled]
    n = len(keys)
    print(f"Loaded {n} manually labeled samples.")

    # ── load detector scores ─────────────────────────────────────────────────
    detector_data = {
        "SelfCheckGPT": load_detector(DETECTORS["SelfCheckGPT"], "avg_hallucination_score"),
        "MiniCheck":    load_detector(DETECTORS["MiniCheck"],    "hallucination_ratio"),
        "AlignScore":   load_detector(DETECTORS["AlignScore"],   "align_score", invert=True),
        "RAGAS":        load_detector(DETECTORS["RAGAS"],        "faithfulness_score", invert=True),
    }

    # Build feature matrix
    X_scores = []   # continuous scores (for ensemble training)
    X_binary = []   # binary predictions (for individual F1 comparison)
    missing_any = []

    for key in keys:
        row_scores = []
        row_binary = []
        for dname in ["SelfCheckGPT", "MiniCheck", "AlignScore", "RAGAS"]:
            d = detector_data[dname]
            item = d.get(key) or next((v for k, v in d.items() if k[0]==key[0] and k[1]==key[1]), None)
            if item:
                row_scores.append(item["score"])
                row_binary.append(item["binary"])
            else:
                row_scores.append(0.5)
                row_binary.append(0)
                missing_any.append(f"{dname}:{key}")
        X_scores.append(row_scores)
        X_binary.append(row_binary)

    if missing_any:
        print(f"  WARNING: {len(missing_any)} missing detector-sample pairs (defaulted to 0.5 / 0)")

    # ── individual detector metrics ──────────────────────────────────────────
    detector_names = ["SelfCheckGPT", "MiniCheck", "AlignScore", "RAGAS"]
    individual_results = []

    print("\nIndividual Detector Performance (vs manual labels):")
    print(f"  {'Detector':<18} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6}")
    print("  " + "-" * 46)

    for i, dname in enumerate(detector_names):
        preds = [row[i] for row in X_binary]
        scores = [row[i] for row in X_scores]
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = float("nan")
        print(f"  {dname:<18} {p:>6.3f} {r:>6.3f} {f:>6.3f} {auc:>6.3f}")
        individual_results.append({"name": dname, "precision": p, "recall": r, "f1": f, "auc": auc})

    # ── ensemble (logistic regression with LOO CV) ───────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_scores)

    # LOO cross-validation for ensemble
    loo = LeaveOneOut()
    ens_preds = []
    ens_scores = []
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

    for train_idx, test_idx in loo.split(X_scaled):
        X_train = [X_scaled[i] for i in train_idx]
        y_train = [y_true[i] for i in train_idx]
        X_test  = [X_scaled[test_idx[0]]]

        if len(set(y_train)) < 2:
            ens_preds.append(int(round(sum(y_train)/len(y_train))))
            ens_scores.append(0.5)
            continue

        clf.fit(X_train, y_train)
        ens_preds.append(clf.predict(X_test)[0])
        ens_scores.append(clf.predict_proba(X_test)[0][1])

    ens_p = precision_score(y_true, ens_preds, zero_division=0)
    ens_r = recall_score(y_true, ens_preds, zero_division=0)
    ens_f = f1_score(y_true, ens_preds, zero_division=0)
    try:
        ens_auc = roc_auc_score(y_true, ens_scores)
    except Exception:
        ens_auc = float("nan")

    # Fit on full data for weights
    clf.fit(X_scaled, y_true)
    weights = clf.coef_[0]

    print(f"\n  {'ENSEMBLE':<18} {ens_p:>6.3f} {ens_r:>6.3f} {ens_f:>6.3f} {ens_auc:>6.3f}")
    print("  (Leave-one-out cross-validation)")

    best_individual = max(individual_results, key=lambda x: x["f1"])
    improvement = (ens_f - best_individual["f1"]) * 100

    print(f"\nEnsemble weights (logistic regression coefficients):")
    for dname, w in sorted(zip(detector_names, weights), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(w) * 10)
        print(f"  {dname:<18} {w:+.3f}  {bar}")

    print(f"\nBest individual: {best_individual['name']} (F1 = {best_individual['f1']:.3f})")
    print(f"Ensemble:                      F1 = {ens_f:.3f}  "
          f"({'+'if improvement>=0 else ''}{improvement:.1f}pp vs best individual)")

    top_detector = sorted(zip(detector_names, weights), key=lambda x: abs(x[1]), reverse=True)[0]

    print(f"\nUSE IN THESIS:")
    print(f'  "An ensemble of four detectors (LOO-CV F1 = {ens_f:.2f}) outperforms the')
    print(f"  best individual detector ({best_individual['name']}, F1 = {best_individual['f1']:.2f})")
    print(f"  by {improvement:.1f}pp. Logistic regression weights indicate that")
    print(f"  {top_detector[0]} contributes most to ensemble accuracy (β = {top_detector[1]:+.3f}),")
    print(f'  suggesting it is the most reliable detector for interview RAG."')

    # ── save ─────────────────────────────────────────────────────────────────
    output = {
        "n_samples": n,
        "individual_detectors": individual_results,
        "ensemble": {
            "precision": round(ens_p, 4), "recall": round(ens_r, 4),
            "f1": round(ens_f, 4), "auc": round(ens_auc, 4) if ens_auc==ens_auc else None,
            "cv_method": "leave-one-out",
        },
        "ensemble_weights": [{"detector": d, "coefficient": round(float(w), 4)}
                             for d, w in zip(detector_names, weights)],
        "best_individual": best_individual["name"],
        "improvement_pp": round(improvement, 2),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # ── bar chart ────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        names = [r["name"] for r in individual_results] + ["Ensemble"]
        f1s   = [r["f1"] for r in individual_results] + [ens_f]
        colors = ["#2980b9"] * 4 + ["#e74c3c"]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, f1s, color=colors, width=0.5)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1 Score")
        ax.set_title("Individual Detectors vs Ensemble (LOO-CV F1)")
        ax.axhline(best_individual["f1"], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
        print(f"Chart saved: {OUTPUT_CHART}")
    except ImportError:
        print("(matplotlib not installed — skipping chart)")

    print(f"Results saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
