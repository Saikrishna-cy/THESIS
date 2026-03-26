"""
Role Attribution Drift Detector
================================
Post-processing step — NO new API calls.
Reads 02_rq1_annotations.json produced by experiment_rq1.py and applies
a heuristic detector for ROLE_ATTRIBUTION_DRIFT patterns.

ROLE_ATTRIBUTION_DRIFT: The model starts with correct speaker attribution but
progressively drifts to wrong speaker across consecutive sentences. This is
distinct from a single-sentence misattribution — it's a sequential positional
pattern grounded in context position decay (arXiv:2108.12409).

Detection logic:
  1. Split response into sentences
  2. Label each sentence: AGENT / PARTICIPANT / AMBIGUOUS using regex on speaker names
  3. Find the first speaker mentioned (sentence 1 = correct anchor)
  4. Check if 3+ of the LAST HALF sentences attribute to a different speaker
  5. If yes: mark as ROLE_ATTRIBUTION_DRIFT

Output:
  - Console: per-model drift counts
  - results/{model}/role_drift_counts.json

Usage:
  python experiments/02_rq1/detect_role_drift.py
  python experiments/02_rq1/detect_role_drift.py --results-base results
  python experiments/02_rq1/detect_role_drift.py --model gpt-4o-mini
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent

# ── speaker patterns ─────────────────────────────────────────────────────────
_AGENT_RE = re.compile(r'\b(agent|interviewer|researcher)\b', re.IGNORECASE)
_PARTICIPANT_RE = re.compile(r'\b(participant|respondent|interviewee)\b', re.IGNORECASE)


def sentence_split(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries (keeps at least 1 word)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def label_sentence(sentence: str) -> str:
    """Return 'AGENT', 'PARTICIPANT', or 'AMBIGUOUS' for a sentence."""
    has_agent = bool(_AGENT_RE.search(sentence))
    has_participant = bool(_PARTICIPANT_RE.search(sentence))
    if has_agent and not has_participant:
        return 'AGENT'
    if has_participant and not has_agent:
        return 'PARTICIPANT'
    return 'AMBIGUOUS'


def has_drift_pattern(response: str, min_drift_sentences: int = 3) -> dict:
    """
    Detect ROLE_ATTRIBUTION_DRIFT in a response string.

    Returns:
        {
            "drift_detected": bool,
            "anchor_speaker": str,         # speaker in sentence 1
            "drift_count": int,            # sentences in latter half with wrong speaker
            "total_sentences": int,
            "latter_labels": list[str],
        }
    """
    sentences = sentence_split(response)
    if len(sentences) < 4:
        return {"drift_detected": False, "anchor_speaker": "AMBIGUOUS",
                "drift_count": 0, "total_sentences": len(sentences), "latter_labels": []}

    labels = [label_sentence(s) for s in sentences]
    # Find anchor: first non-AMBIGUOUS label
    anchor = "AMBIGUOUS"
    for lbl in labels:
        if lbl != "AMBIGUOUS":
            anchor = lbl
            break

    if anchor == "AMBIGUOUS":
        return {"drift_detected": False, "anchor_speaker": "AMBIGUOUS",
                "drift_count": 0, "total_sentences": len(sentences), "latter_labels": labels}

    # Check latter half (second half of sentences)
    half = len(sentences) // 2
    latter_labels = labels[half:]
    opposite = "PARTICIPANT" if anchor == "AGENT" else "AGENT"
    drift_count = sum(1 for lbl in latter_labels if lbl == opposite)

    drift_detected = drift_count >= min_drift_sentences

    return {
        "drift_detected": drift_detected,
        "anchor_speaker": anchor,
        "drift_count": drift_count,
        "total_sentences": len(sentences),
        "latter_labels": latter_labels,
    }


def detect_drift_in_results(annotations: list[dict]) -> dict:
    """
    Run drift detection over all annotations from 02_rq1_annotations.json.

    Returns summary dict with per-item results and aggregate counts.
    """
    results = []
    drift_count = 0
    total = 0

    for ann in annotations:
        response = ann.get("rag_response", "")
        if not response:
            continue
        total += 1
        detection = has_drift_pattern(response)
        if detection["drift_detected"]:
            drift_count += 1
        results.append({
            "interview_id": ann.get("interview_id", ""),
            "query_type": ann.get("query_type", ""),
            "rag_model": ann.get("rag_model", "unknown"),
            "language": ann.get("language", ""),
            "drift_detected": detection["drift_detected"],
            "anchor_speaker": detection["anchor_speaker"],
            "drift_count": detection["drift_count"],
            "total_sentences": detection["total_sentences"],
        })

    drift_rate = drift_count / total if total > 0 else 0.0
    return {
        "total_responses": total,
        "drift_detected_count": drift_count,
        "drift_rate": round(drift_rate, 4),
        "per_response": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect ROLE_ATTRIBUTION_DRIFT in annotations")
    parser.add_argument("--results-base", default=str(_ROOT / "results"),
                        help="Base results directory")
    parser.add_argument("--model", default=None,
                        help="Single model to process (default: all model subdirs)")
    args = parser.parse_args()

    results_base = Path(args.results_base)
    if not results_base.exists():
        print(f"ERROR: results base not found: {results_base}")
        sys.exit(1)

    # Find model dirs
    if args.model:
        model_dirs = [results_base / args.model]
    else:
        model_dirs = [d for d in results_base.iterdir() if d.is_dir()]

    print("=" * 60)
    print("ROLE ATTRIBUTION DRIFT DETECTOR")
    print("Theory: positional decay (arXiv:2108.12409)")
    print("=" * 60)

    overall = {}
    for model_dir in sorted(model_dirs):
        ann_file = model_dir / "02_rq1_annotations.json"
        if not ann_file.exists():
            print(f"  [SKIP] {model_dir.name} — no 02_rq1_annotations.json")
            continue

        with open(ann_file, encoding="utf-8") as f:
            annotations = json.load(f)

        summary = detect_drift_in_results(annotations)
        model_name = model_dir.name

        print(f"\n{model_name}:")
        print(f"  Total responses: {summary['total_responses']}")
        print(f"  Drift detected:  {summary['drift_detected_count']} "
              f"({summary['drift_rate']:.1%})")

        out_path = model_dir / "role_drift_counts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved → {out_path}")

        overall[model_name] = {
            "total": summary["total_responses"],
            "drift_count": summary["drift_detected_count"],
            "drift_rate": summary["drift_rate"],
        }

    print("\n" + "=" * 60)
    print("SUMMARY — ROLE_ATTRIBUTION_DRIFT by model:")
    print(f"  {'Model':<20} {'Total':>8} {'Drift':>8} {'Rate':>8}")
    print("  " + "-" * 46)
    for model, stats in sorted(overall.items()):
        print(f"  {model:<20} {stats['total']:>8} "
              f"{stats['drift_count']:>8} {stats['drift_rate']:>7.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
