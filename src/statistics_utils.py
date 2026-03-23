"""
Statistics Utilities for IEEE Publication
==========================================
Provides significance tests and confidence intervals needed for rigorous
reporting in the IEEE paper.

Functions:
  mcnemar_test(labels_a, labels_b)      — pairwise model comparison
  confidence_interval(p, n, z=1.96)     — Wilson score CI for proportions
  compare_models(unified_jsonl)          — full pairwise model comparison table
  detector_confidence_intervals(results) — CI for Precision/Recall/F1

Usage:
  python src/statistics_utils.py --input data/unified_results.jsonl
"""

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent


# ── Core statistical functions ────────────────────────────────────────────────

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) as floats [0, 1].

    Better than normal approximation for small n or extreme proportions.
    """
    if n == 0:
        return (0.0, 1.0)
    denominator = 1 + z**2 / n
    centre      = (p + z**2 / (2 * n)) / denominator
    margin      = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_test(labels_a: list, labels_b: list) -> dict:
    """
    McNemar's test for paired nominal data.
    Tests whether two classifiers make significantly different errors on the
    same samples. Appropriate for comparing model-level hallucination rates
    on the same interview set.

    labels_a, labels_b: lists of bool (True = hallucinated)

    Returns:
        {
          "b": int,           # A correct, B wrong
          "c": int,           # A wrong, B correct
          "statistic": float, # chi-square statistic (with continuity correction)
          "p_value": float,   # p-value (two-tailed)
          "significant": bool # p < 0.05
        }

    Reference: McNemar (1947). Note on the sampling error of the difference
    between correlated proportions. Psychometrika, 12(2), 153–157.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")

    b = sum(1 for a, bb in zip(labels_a, labels_b) if a and not bb)   # A=hall, B=ok
    c = sum(1 for a, bb in zip(labels_a, labels_b) if not a and bb)   # A=ok,  B=hall

    if (b + c) == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "significant": False}

    # Continuity-corrected McNemar statistic
    stat = (abs(b - c) - 1) ** 2 / (b + c)

    # Chi-square with 1 df — p-value via survival function approximation
    # We implement a simple chi-square p-value without scipy
    p_value = _chi2_sf(stat, df=1)

    return {
        "b":           b,
        "c":           c,
        "statistic":   round(stat, 4),
        "p_value":     round(p_value, 4),
        "significant": p_value < 0.05,
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """
    Survival function (1 - CDF) for chi-square distribution with df degrees of freedom.
    Implemented via regularized incomplete gamma function approximation.
    For df=1 (our use case) this is equivalent to 2 * Phi(-sqrt(x)).
    """
    if df == 1:
        # P(chi2 > x) = 2 * (1 - Phi(sqrt(x))) = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(x / 2))
    # General case via series expansion (good enough for our purposes)
    # Uses the regularized upper incomplete gamma function Q(a, x) with a = df/2, x = x/2
    a = df / 2.0
    xx = x / 2.0
    return _upper_incomplete_gamma(a, xx)


def _upper_incomplete_gamma(a: float, x: float, max_iter: int = 200) -> float:
    """Q(a, x) = 1 - P(a, x). Series expansion for small x, continued fraction for large x."""
    if x == 0:
        return 1.0
    if x < a + 1:
        # Series representation
        term = math.exp(-x + a * math.log(x) - _log_gamma(a))
        total = 1.0 / a
        factor = 1.0 / a
        for n in range(1, max_iter):
            factor *= x / (a + n)
            total += factor
            if abs(factor) < 1e-12 * abs(total):
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - _log_gamma(a))
    else:
        # Continued fraction (Lentz)
        fpmin = 1e-30
        b = x + 1 - a
        c = 1 / fpmin
        d = 1 / b
        h = d
        for i in range(1, max_iter):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < fpmin:
                d = fpmin
            c = b + an / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-12:
                break
        return math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def _log_gamma(x: float) -> float:
    """Lanczos approximation for log(Gamma(x))."""
    g = 7
    coeffs = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1 - x)
    x -= 1
    a = coeffs[0]
    t = x + g + 0.5
    for i, c in enumerate(coeffs[1:], 1):
        a += c / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


# ── Model comparison table ─────────────────────────────────────────────────────

def compare_models(unified_jsonl: Path) -> dict:
    """
    Load unified_results.jsonl and compute pairwise McNemar tests between all models.
    Also computes per-model hallucination rate with 95% Wilson CIs.

    Returns:
        {
          "per_model": {
            "gpt-4o-mini": {"n": int, "rate": float, "ci_low": float, "ci_high": float},
            ...
          },
          "pairwise": {
            "gpt-4o-mini vs qwen": {mcnemar result dict},
            ...
          }
        }
    """
    # Load
    records = []
    with open(unified_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Group by model, keyed by query_id for alignment
    by_model = defaultdict(dict)
    for r in records:
        model = r.get("model", "unknown")
        qid   = r.get("query_id", "")
        label = r.get("rq1_label", "") == "HALLUCINATED"
        by_model[model][qid] = label

    models   = sorted(by_model.keys())
    per_model = {}

    for m in models:
        labels = list(by_model[m].values())
        n    = len(labels)
        rate = sum(labels) / n if n > 0 else 0.0
        ci   = wilson_ci(rate, n)
        per_model[m] = {
            "n":       n,
            "rate":    round(rate, 4),
            "ci_low":  round(ci[0], 4),
            "ci_high": round(ci[1], 4),
        }

    # Pairwise McNemar — align on shared query_ids
    pairwise = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            shared = sorted(set(by_model[m1]) & set(by_model[m2]))
            if len(shared) < 10:
                pairwise[f"{m1} vs {m2}"] = {"error": "insufficient shared samples"}
                continue
            la = [by_model[m1][k] for k in shared]
            lb = [by_model[m2][k] for k in shared]
            result = mcnemar_test(la, lb)
            result["n_shared"] = len(shared)
            pairwise[f"{m1} vs {m2}"] = result

    return {"per_model": per_model, "pairwise": pairwise}


# ── Detector CI table ──────────────────────────────────────────────────────────

def detector_confidence_intervals(unified_jsonl: Path, manual_labels_csv: Path = None) -> dict:
    """
    Compute Precision/Recall/F1 with 95% Wilson CIs for each detector.

    Uses RQ1 label as ground truth. Detector threshold = 0.5 on score.
    If manual_labels_csv is provided, uses human labels instead.

    Returns per-model, per-detector metrics dict.
    """
    import csv

    records = []
    with open(unified_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Optional: override ground truth with manual labels
    manual_override = {}
    if manual_labels_csv and Path(manual_labels_csv).exists():
        with open(manual_labels_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("YOUR_label", "").strip() in ("HALLUCINATED", "FAITHFUL"):
                    key = f"{row.get('interview_id', '')}___{row.get('query_type', '')}"
                    manual_override[key] = row["YOUR_label"].strip() == "HALLUCINATED"

    detectors = {
        "SelfCheckGPT": ("selfcheck_score",   0.5),
        "MiniCheck":    ("minicheck_score",   0.5),
        "AlignScore":   ("alignscore",        0.5),
        "RAGAS":        ("ragas_faithfulness", 0.5),
    }

    results = {}

    models = sorted({r.get("model", "") for r in records})
    for model in models:
        results[model] = {}
        model_recs = [r for r in records if r.get("model") == model]

        for det_name, (field, threshold) in detectors.items():
            valid = [r for r in model_recs if r.get(field) is not None]
            if not valid:
                results[model][det_name] = {"status": "pending"}
                continue

            tp = fp = tn = fn = 0
            for r in valid:
                qid       = r.get("query_id", "")
                if qid in manual_override:
                    gt = manual_override[qid]
                else:
                    gt = r.get("rq1_label", "") == "HALLUCINATED"

                score  = r[field]
                # AlignScore: higher = more faithful → invert
                if field == "alignscore":
                    pred = score < threshold
                else:
                    pred = score >= threshold

                if gt and pred:     tp += 1
                elif not gt and pred: fp += 1
                elif not gt and not pred: tn += 1
                else: fn += 1

            n = tp + fp + tn + fn
            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            prec_ci = wilson_ci(prec, tp + fp)
            rec_ci  = wilson_ci(rec,  tp + fn)
            f1_ci   = wilson_ci(f1,   n)

            results[model][det_name] = {
                "n":             n,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision":     round(prec, 4),
                "precision_ci":  (round(prec_ci[0], 4), round(prec_ci[1], 4)),
                "recall":        round(rec,  4),
                "recall_ci":     (round(rec_ci[0],  4), round(rec_ci[1],  4)),
                "f1":            round(f1,   4),
                "f1_ci":         (round(f1_ci[0],   4), round(f1_ci[1],   4)),
            }

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute statistical tests for thesis")
    parser.add_argument("--input", default=str(BASE / "data" / "unified_results.jsonl"),
                        help="Path to unified_results.jsonl")
    parser.add_argument("--manual-labels",
                        default=str(BASE / "RAG_THESIS" / "output" / "manual_validation_200.csv"),
                        help="Path to manually annotated CSV (optional)")
    parser.add_argument("--output", default=str(BASE / "RAG_THESIS" / "output" / "statistics_report.json"))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run build_unified_results.py first.")
        raise SystemExit(1)

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Model comparison
    print("\n1. Per-model hallucination rates with 95% CI:")
    comparison = compare_models(input_path)
    for model, stats in comparison["per_model"].items():
        print(f"  {model:<20} rate={stats['rate']:.1%}  "
              f"95% CI [{stats['ci_low']:.1%}, {stats['ci_high']:.1%}]  n={stats['n']}")

    print("\n2. Pairwise McNemar tests:")
    for pair, result in comparison["pairwise"].items():
        if "error" in result:
            print(f"  {pair}: {result['error']}")
        else:
            sig = "SIGNIFICANT" if result["significant"] else "not significant"
            print(f"  {pair}: χ²={result['statistic']:.3f}, p={result['p_value']:.4f} → {sig}")

    # Detector CIs
    manual = args.manual_labels if Path(args.manual_labels).exists() else None
    print("\n3. Detector performance with 95% CI (using RQ1 as ground truth):")
    det_results = detector_confidence_intervals(input_path, manual)
    for model, detectors in det_results.items():
        print(f"\n  Model: {model}")
        for det, stats in detectors.items():
            if stats.get("status") == "pending":
                print(f"    {det:<15} [pending]")
            else:
                f1_lo, f1_hi = stats["f1_ci"]
                print(f"    {det:<15} F1={stats['f1']:.3f} [{f1_lo:.3f}–{f1_hi:.3f}]  "
                      f"P={stats['precision']:.3f}  R={stats['recall']:.3f}")

    # Save
    output = {
        "model_comparison": comparison,
        "detector_metrics": det_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
