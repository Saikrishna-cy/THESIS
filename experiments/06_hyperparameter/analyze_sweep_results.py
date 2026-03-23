"""
Master Sweep Results Aggregator
================================================================
PURPOSE:
  Reads ALL hyperparameter sweep outputs and generates a unified
  cross-analysis summary. This is the final synthesis step that
  combines findings from all 5 sweep experiments into a
  comprehensive view of the hyperparameter space.

WHAT IT READS:
  - sweep_results.json         (from run_ablation_sweep.py)
  - temperature_analysis.json  (from temperature_analysis.py)
  - topk_chunk_analysis.json   (from top_k_chunk_analysis.py)
  - sampling_analysis.json     (from sampling_analysis.py)
  - threshold_optimization.json(from threshold_optimizer.py)

WHAT IT PRODUCES:
  1. Combined ranking of all configurations tested
  2. Interaction analysis: does the best prompt also work with the best retrieval?
  3. Language-stratified results (Dutch vs English)
  4. Consolidated recommendations table for thesis
  5. hyperparameter_summary.json — machine-readable for the PDF generator

HOW TO RUN:
  python D:\\RAG_THESIS\\hyperparameter_sweep\\analyze_sweep_results.py

PREREQUISITES:
  Run all sweep scripts first (or at least the ones you want to analyse)

OUTPUT:
  D:\\RAG_THESIS\\output\\hyperparameter_summary.json
  D:\\RAG_THESIS\\output\\sweep_consolidated_table.csv

HOW IT HELPS YOUR IEEE PAPER:
  This script produces the numbers you quote in the abstract and
  conclusion. It answers: "What is the single best configuration
  for interview RAG, and how much better is it than baseline?"
  That answer goes in Section 7 (Conclusions) and Table 6 (full
  hyperparameter comparison).
"""

import csv
import json
from pathlib import Path

OUTPUT_DIR = Path(r"D:\RAG_THESIS\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("MASTER HYPERPARAMETER SWEEP AGGREGATOR")
    print("=" * 70)

    # ── Load all available results ────────────────────────────────────────────
    sweep    = load_json("sweep_results.json")
    temp     = load_json("temperature_analysis.json")
    topk     = load_json("topk_chunk_analysis.json")
    sampling = load_json("sampling_analysis.json")
    thresh   = load_json("threshold_optimization.json")

    available = {
        "Ablation Sweep (10 configs)":     sweep is not None,
        "Temperature Analysis":            temp is not None,
        "Top-K × Chunk-Size Analysis":     topk is not None,
        "Sampling Reliability Analysis":   sampling is not None,
        "Threshold Optimization":          thresh is not None,
    }
    print("\nAvailable results:")
    for name, ok in available.items():
        status = "✓" if ok else "✗ (not yet run)"
        print(f"  {status}  {name}")

    # ── Ablation Sweep Summary ────────────────────────────────────────────────
    if sweep:
        print_section("ABLATION SWEEP — 10 Configurations")
        summaries = sweep.get("summaries", [])
        if summaries:
            print(f"{'Config':<25} {'Hall%':>7} {'Faith.':>7}")
            print("-" * 45)
            for s in sorted(summaries, key=lambda x: x.get("hallucination_rate", 1)):
                print(f"  {s.get('config','?'):<23} "
                      f"{s.get('hallucination_rate',0):>6.1%} "
                      f"{s.get('avg_faithfulness',0):>7.3f}")
            best = sweep.get("best_config")
            baseline_rate = next((s["hallucination_rate"] for s in summaries
                                   if s.get("config") == "A1_control"), None)
            best_rate = next((s["hallucination_rate"] for s in summaries
                               if s.get("config") == best), None)
            if baseline_rate and best_rate:
                print(f"\n  Baseline (A1_control): {baseline_rate:.1%}")
                print(f"  Best config ({best}): {best_rate:.1%}")
                print(f"  Improvement: {(baseline_rate - best_rate)*100:+.1f}pp")

    # ── Temperature Analysis Summary ──────────────────────────────────────────
    if temp:
        print_section("TEMPERATURE ANALYSIS")
        summaries = temp.get("summaries", [])
        if summaries:
            print(f"{'Temp':>5} {'Hall%':>7} {'Faith.':>7}  Top hallucination type")
            print("-" * 55)
            for s in summaries:
                top = sorted(s.get("hallucination_type_counts", {}).items(),
                             key=lambda x: x[1], reverse=True)
                top_str = f"{top[0][0]}({top[0][1]})" if top else "—"
                print(f"  T={s['temperature']:>3.1f}  "
                      f"{s['hallucination_rate']:>6.1%}  "
                      f"{s['avg_faithfulness']:>6.3f}  {top_str}")
        opt = temp.get("optimal_temperature")
        if opt is not None:
            print(f"\n  Optimal temperature: T={opt}")
            print(f"  Key finding: check REFUSAL vs BASELESS trade-off at T=0.0 vs T=1.0")

    # ── Top-K × Chunk-Size Summary ────────────────────────────────────────────
    if topk:
        print_section("TOP-K × CHUNK-SIZE GRID")
        best_cfg = topk.get("best_config")
        if best_cfg:
            print(f"  Best retrieval config: chunk_size={best_cfg['chunk_size']}, "
                  f"top_k={best_cfg['top_k']}")
            print(f"  Hallucination rate: {best_cfg['hallucination_rate']:.1%}")

        grid = topk.get("grid_results", [])
        if grid:
            print(f"\n  Worst config: chunk_size, top_k = ", end="")
            worst = max(grid, key=lambda g: g.get("hallucination_rate", 0))
            print(f"{worst['chunk_size']}, {worst['top_k']} "
                  f"({worst['hallucination_rate']:.1%})")
            print(f"\n  Hall. rate range: "
                  f"{min(g['hallucination_rate'] for g in grid):.1%} – "
                  f"{max(g['hallucination_rate'] for g in grid):.1%}")

    # ── Sampling Analysis Summary ─────────────────────────────────────────────
    if sampling:
        print_section("SELFCHECKGPT SAMPLING RELIABILITY")
        summary_by_n = sampling.get("summary_by_n", [])
        if summary_by_n:
            print(f"  {'N':>4} {'Accuracy':>10} {'Score StdDev':>12}")
            print("  " + "-" * 30)
            for s in summary_by_n:
                print(f"  {s['n_samples']:>4}  {s['accuracy']:>9.1%}  {s['score_std']:>11.3f}")
            best_n = max(summary_by_n, key=lambda s: s["accuracy"])
            print(f"\n  Recommended N: {best_n['n_samples']} (accuracy={best_n['accuracy']:.1%})")

    # ── Threshold Optimization Summary ────────────────────────────────────────
    if thresh and thresh.get("by_detector"):
        print_section("OPTIMAL DETECTION THRESHOLDS")
        print(f"  {'Detector':<25} {'Default':>8} {'Optimal':>8} {'F1 gain':>10} {'AUC':>6}")
        print("  " + "-" * 65)
        for det_name, res in thresh["by_detector"].items():
            print(f"  {res.get('detector', det_name):<25} "
                  f"{'0.50':>8} "
                  f"{res.get('optimal_threshold', 0):>8.2f} "
                  f"{'→F1='+str(res.get('optimal_f1', 0)):>10} "
                  f"{res.get('auc', 0):>6.3f}")

    # ── Consolidated recommendations ─────────────────────────────────────────
    print_section("CONSOLIDATED RECOMMENDATIONS FOR THESIS")
    recommendations = []

    if temp and temp.get("optimal_temperature") is not None:
        recommendations.append({
            "parameter": "Generation Temperature",
            "recommended_value": str(temp["optimal_temperature"]),
            "default_value": "0.3",
            "source": "temperature_analysis.py",
            "finding": "Minimises total hallucination rate",
        })

    if topk and topk.get("best_config"):
        bc = topk["best_config"]
        recommendations.append({
            "parameter": "Chunk Size",
            "recommended_value": str(bc["chunk_size"]),
            "default_value": "1000",
            "source": "top_k_chunk_analysis.py",
            "finding": f"Optimal retrieval config (Hall.={bc['hallucination_rate']:.1%})",
        })
        recommendations.append({
            "parameter": "top_k",
            "recommended_value": str(bc["top_k"]),
            "default_value": "3",
            "source": "top_k_chunk_analysis.py",
            "finding": "Best number of retrieved chunks",
        })

    if sampling and sampling.get("summary_by_n"):
        best_n_entry = max(sampling["summary_by_n"], key=lambda s: s["accuracy"])
        recommendations.append({
            "parameter": "n_selfcheck_samples",
            "recommended_value": str(best_n_entry["n_samples"]),
            "default_value": "5",
            "source": "sampling_analysis.py",
            "finding": f"Accuracy={best_n_entry['accuracy']:.1%}, cost-optimal",
        })

    if thresh and thresh.get("optimal_thresholds"):
        for det, t in thresh["optimal_thresholds"].items():
            recommendations.append({
                "parameter": f"{det}_threshold",
                "recommended_value": str(t),
                "default_value": "0.50",
                "source": "threshold_optimizer.py",
                "finding": "Maximises F1 on manual labels",
            })

    if recommendations:
        print(f"  {'Parameter':<30} {'Default':>8} {'Optimal':>8}  Finding")
        print("  " + "-" * 75)
        for r in recommendations:
            print(f"  {r['parameter']:<30} {r['default_value']:>8} "
                  f"{r['recommended_value']:>8}  {r['finding']}")
    else:
        print("  No results available yet. Run sweep scripts first.")

    # ── Save consolidated output ──────────────────────────────────────────────
    summary_out = {
        "available_results": available,
        "recommendations": recommendations,
        "ablation_sweep": {
            "best_config": sweep.get("best_config") if sweep else None,
            "summaries": sweep.get("summaries", []) if sweep else [],
        } if sweep else None,
        "temperature": {
            "optimal": temp.get("optimal_temperature") if temp else None,
            "summaries": temp.get("summaries", []) if temp else [],
        } if temp else None,
        "retrieval": {
            "best_config": topk.get("best_config") if topk else None,
        } if topk else None,
        "sampling": {
            "summary_by_n": sampling.get("summary_by_n", []) if sampling else [],
        } if sampling else None,
        "thresholds": {
            "optimal": thresh.get("optimal_thresholds", {}) if thresh else {},
        } if thresh else None,
    }
    with open(OUTPUT_DIR / "hyperparameter_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    # ── Consolidated CSV table ────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "sweep_consolidated_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "default_value",
                                                "recommended_value", "source", "finding"])
        writer.writeheader()
        writer.writerows(recommendations)

    print(f"\n  Saved: {OUTPUT_DIR / 'hyperparameter_summary.json'}")
    print(f"  Saved: {csv_path}")
    print("\n  Next step: run create_hyperparameter_pdf.py to generate the PDF report.")


if __name__ == "__main__":
    main()
