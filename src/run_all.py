"""
Master Orchestrator
====================
Runs the full hallucination research pipeline for one or more models.

Usage examples:
  # Full pipeline, all 3 models, all data:
  python src/run_all.py --models gpt-4o-mini,qwen,mistral --steps all

  # Generate only, limit to 5 interviews for testing:
  python src/run_all.py --models gpt-4o-mini --steps generate --max-interviews 5

  # RQ1 + RQ2 only (responses already generated):
  python src/run_all.py --models qwen,mistral --steps rq1,rq2

  # Cross-model comparison only:
  python src/run_all.py --steps compare
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path so sibling imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

# ── default data sources ────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent

DEFAULT_DATA_SOURCES = [
    str(_BASE / "data" / "real"      / "supabase_responses.csv"),
    str(_BASE / "data" / "synthetic" / "synthetic_interviews.csv"),
    str(_BASE / "data" / "synthetic" / "supbase_english_synthetic.csv"),
    str(_BASE / "data" / "synthetic" / "supbase_dutch_synthetic.csv"),
]

DEFAULT_MODELS   = ["gpt-4o-mini", "qwen", "mistral"]
DEFAULT_RESULTS  = str(_BASE / "results")


# ── helpers ─────────────────────────────────────────────────────────────────

def results_dir(model: str, base: str = DEFAULT_RESULTS) -> str:
    """Map model name → per-model results subdirectory."""
    safe = model.replace("/", "-").replace(".", "-")
    return str(Path(base) / safe)


def _header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ── pipeline steps ───────────────────────────────────────────────────────────

def step_generate(models, data_sources, max_interviews, results_base,
                  selfcheck_samples=5):
    """Step 1 — generate RAG responses for each model."""
    from data_loader import merge_csv_sources, prepare_all_samples
    from rag_pipeline import generate_all_responses

    _header("STEP: GENERATE RAG RESPONSES")

    existing = [p for p in data_sources if Path(p).exists()]
    missing  = [p for p in data_sources if not Path(p).exists()]
    if missing:
        print(f"WARNING: {len(missing)} data source(s) not found — skipping:")
        for m in missing:
            print(f"  {m}")

    if not existing:
        print("ERROR: No valid data sources found. Aborting generate step.")
        return

    interviews = merge_csv_sources(existing)
    samples    = prepare_all_samples(interviews, max_interviews=max_interviews)
    print(f"Total samples: {len(samples)}")

    for model in models:
        rdir = results_dir(model, results_base)
        _header(f"Generating responses — model: {model}")
        generate_all_responses(
            samples,
            model=model,
            include_selfcheck_samples=True,
            n_selfcheck_samples=selfcheck_samples,
            results_dir=rdir,
        )


def step_rq1(models, results_base):
    """Step 2 — RQ1 hallucination taxonomy experiments."""
    from experiment_rq1 import run_all_rq1

    _header("STEP: RQ1 — HALLUCINATION TAXONOMY")
    for model in models:
        rdir = results_dir(model, results_base)
        _header(f"RQ1 — model: {model}")
        run_all_rq1(results_dir=rdir)


def step_rq2(models, results_base):
    """Step 3 — RQ2 hallucination detection experiments."""
    from experiment_rq2 import run_all_rq2

    _header("STEP: RQ2 — HALLUCINATION DETECTION")
    for model in models:
        rdir = results_dir(model, results_base)
        _header(f"RQ2 — model: {model}")
        run_all_rq2(results_dir=rdir)


def step_compare(results_base):
    """Step 4 — cross-model comparison report."""
    from compare_models import generate_comparison_report

    _header("STEP: CROSS-MODEL COMPARISON")
    comparison_dir = str(Path(results_base) / "comparison")
    generate_comparison_report(
        results_base=results_base,
        output_dir=comparison_dir,
    )


def step_unified(results_base):
    """Step 5 — build unified_results.jsonl for advanced modules."""
    from build_unified_results import build

    _header("STEP: BUILD UNIFIED RESULTS")
    output_path = _BASE / "data" / "unified_results.jsonl"
    build(Path(results_base), output_path)
    print(f"Unified results written → {output_path}")


def step_stats(results_base):
    """Step 6 — statistical significance tests + CIs."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from statistics_utils import compare_models, detector_confidence_intervals
    import json

    _header("STEP: STATISTICAL ANALYSIS")
    unified = _BASE / "data" / "unified_results.jsonl"
    if not unified.exists():
        print("WARNING: unified_results.jsonl not found. Run --steps unified first.")
        return

    comparison = compare_models(unified)
    print("\nPer-model hallucination rates with 95% CI:")
    for model, stats in comparison["per_model"].items():
        print(f"  {model:<20} rate={stats['rate']:.1%}  "
              f"CI [{stats['ci_low']:.1%}, {stats['ci_high']:.1%}]")

    print("\nPairwise McNemar tests:")
    for pair, result in comparison["pairwise"].items():
        if "error" in result:
            print(f"  {pair}: {result['error']}")
        else:
            sig = "SIGNIFICANT" if result["significant"] else "not significant"
            print(f"  {pair}: χ²={result['statistic']:.3f}, p={result['p_value']:.4f} → {sig}")

    out = _BASE / "RAG_THESIS" / "output" / "statistics_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model_comparison": comparison}, f, indent=2)
    print(f"\nSaved → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run hallucination research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names (default: gpt-4o-mini,qwen,mistral)",
    )
    parser.add_argument(
        "--data-sources",
        default=",".join(DEFAULT_DATA_SOURCES),
        help="Comma-separated CSV paths (default: all 4 data files)",
    )
    parser.add_argument(
        "--max-interviews",
        type=int,
        default=None,
        help="Limit number of interviews (default: no limit)",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Which steps to run: all | generate | rq1 | rq2 | compare | unified | stats "
             "(comma-separated, e.g. rq1,rq2 or unified,stats)",
    )
    parser.add_argument(
        "--results-base",
        default=DEFAULT_RESULTS,
        help=f"Base directory for results (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--selfcheck-samples",
        type=int,
        default=5,
        help="Number of SelfCheckGPT samples per response (default: 5)",
    )

    args = parser.parse_args()

    models       = [m.strip() for m in args.models.split(",") if m.strip()]
    data_sources = [p.strip() for p in args.data_sources.split(",") if p.strip()]
    steps_raw    = args.steps.lower()

    if steps_raw == "all":
        steps = ["generate", "rq1", "rq2", "compare", "unified", "stats"]
    else:
        steps = [s.strip() for s in steps_raw.split(",") if s.strip()]

    print("\n" + "=" * 60)
    print("THESIS HALLUCINATION RESEARCH PIPELINE")
    print("=" * 60)
    print(f"  Models       : {models}")
    print(f"  Data sources : {len(data_sources)} file(s)")
    print(f"  Max interviews: {args.max_interviews or 'no limit'}")
    print(f"  Steps        : {steps}")
    print(f"  Results base : {args.results_base}")

    if "generate" in steps:
        step_generate(
            models, data_sources,
            max_interviews=args.max_interviews,
            results_base=args.results_base,
            selfcheck_samples=args.selfcheck_samples,
        )

    if "rq1" in steps:
        step_rq1(models, args.results_base)

    if "rq2" in steps:
        step_rq2(models, args.results_base)

    if "compare" in steps:
        step_compare(args.results_base)

    if "unified" in steps:
        step_unified(args.results_base)

    if "stats" in steps:
        step_stats(args.results_base)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
