"""
Cross-Model Comparison
=======================
Reads RQ1 + RQ2 results from all 3 model subdirs and produces:
  - results/comparison/model_comparison.json   (raw numbers)
  - results/comparison/comparison_report.pdf   (ReportLab tables + charts)

Usage:
  python src/compare_models.py                        # uses default paths
  python src/compare_models.py --results-base results
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ── constants ────────────────────────────────────────────────────────────────

MODELS = ["gpt-4o-mini", "qwen", "mistral"]

# 7 RQ1 hallucination types
HAL_TYPES = [
    "EVIDENT_CONFLICT",
    "SUBTLE_CONFLICT",
    "BASELESS_INFO",
    "SPEAKER_MISATTRIBUTION",
    "TEMPORAL_CONFUSION",
    "SENTIMENT_MISREPRESENTATION",
    "REFUSAL_HALLUCINATION",
]

# 6 DiaHaLu dialogue issue types
DIAL_TYPES = [
    "CROSS_TURN_CONTRADICTION",
    "ENTITY_CONFUSION",
    "TOPIC_DRIFT",
    "TEMPORAL_DISTORTION",
    "CONTEXT_FABRICATION",
    "SPEAKER_CONFUSION",
]

# 4 RQ2 detection methods (file → key in comparison dict)
DETECTION_METHODS = {
    "05_rq2_selfcheck.json":  "selfcheck",
    "06_rq2_minicheck.json":  "minicheck",
    "07_rq2_alignscore.json": "alignscore",
    "08_rq2_ragas.json":      "ragas",
}

QUERY_TYPES = [
    "speaker_attribution",
    "participant_content",
    "factual_summary",
    "specific_content",
    "sentiment",
    "temporal",
]

# ── data loading ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model_results(results_base: str) -> Dict:
    """
    Load all available result files for every model.
    Returns:
      {
        "gpt-4o-mini": {
          "annotations": [...],   # 02_rq1_annotations.json
          "diahalu":     {...},   # 04_rq1_diahalu_eval.json
          "selfcheck":   [...],   # 05_rq2_selfcheck.json
          "minicheck":   [...],   # 06_rq2_minicheck.json
          "alignscore":  [...],   # 07_rq2_alignscore.json
        },
        ...
      }
    """
    base = Path(results_base)
    results = {}
    for model in MODELS:
        mdir = base / model
        if not mdir.exists():
            print(f"  [SKIP] {model} — no results directory found")
            continue
        entry = {}
        entry["annotations"] = _load_json(mdir / "02_rq1_annotations.json") or []
        diahalu_raw = _load_json(mdir / "04_rq1_diahalu_eval.json")
        entry["diahalu"] = diahalu_raw.get("evaluations", []) if isinstance(diahalu_raw, dict) else []
        for fname, key in DETECTION_METHODS.items():
            raw = _load_json(mdir / fname)
            entry[key] = raw if isinstance(raw, list) else []
        results[model] = entry
        print(f"  [{model}] annotations={len(entry['annotations'])} "
              f"diahalu={len(entry['diahalu'])} "
              f"selfcheck={len(entry['selfcheck'])} "
              f"minicheck={len(entry['minicheck'])} "
              f"alignscore={len(entry['alignscore'])} "
              f"ragas={len(entry['ragas'])}")
    return results


# ── computation ───────────────────────────────────────────────────────────────

def _pct(num, denom) -> float:
    return round(100 * num / denom, 1) if denom else 0.0


def compute_hallucination_rates(results: Dict) -> Dict:
    """Per-model hallucination rate overall and per type."""
    out = {}
    for model, data in results.items():
        anns = data["annotations"]
        total = len(anns)
        hallucinated = sum(1 for a in anns if a.get("overall_label") == "HALLUCINATED")
        faithful     = total - hallucinated

        type_counts = defaultdict(int)
        for a in anns:
            for ht in a.get("hallucination_types", []):
                type_counts[ht] += 1

        by_query = defaultdict(lambda: {"total": 0, "hallucinated": 0})
        for a in anns:
            qt = a.get("query_type", "unknown")
            by_query[qt]["total"] += 1
            if a.get("overall_label") == "HALLUCINATED":
                by_query[qt]["hallucinated"] += 1

        by_lang = defaultdict(lambda: {"total": 0, "hallucinated": 0})
        for a in anns:
            lang = a.get("language", "unknown")
            by_lang[lang]["total"] += 1
            if a.get("overall_label") == "HALLUCINATED":
                by_lang[lang]["hallucinated"] += 1

        avg_score = (
            round(sum(a.get("faithfulness_score", 0) for a in anns if a.get("faithfulness_score", -1) >= 0)
                  / max(1, sum(1 for a in anns if a.get("faithfulness_score", -1) >= 0)), 3)
        )

        out[model] = {
            "total": total,
            "hallucinated": hallucinated,
            "faithful": faithful,
            "hallucination_rate_pct": _pct(hallucinated, total),
            "avg_faithfulness_score": avg_score,
            "type_counts": dict(type_counts),
            "by_query_type": {k: {"total": v["total"],
                                   "hallucinated": v["hallucinated"],
                                   "rate_pct": _pct(v["hallucinated"], v["total"])}
                               for k, v in by_query.items()},
            "by_language": {k: {"total": v["total"],
                                 "hallucinated": v["hallucinated"],
                                 "rate_pct": _pct(v["hallucinated"], v["total"])}
                             for k, v in by_lang.items()},
        }
    return out


def compute_diahalu_rates(results: Dict) -> Dict:
    """Per-model DiaHaLu dialogue issue breakdown."""
    out = {}
    for model, data in results.items():
        evals = data["diahalu"]
        total = len(evals)
        faithful = sum(1 for e in evals if e.get("is_faithful"))
        issue_counts = defaultdict(int)
        for e in evals:
            for it in e.get("issue_types", []):
                issue_counts[it] += 1
        avg_score = (
            round(sum(e.get("faithfulness_score", 0) for e in evals if e.get("faithfulness_score", -1) >= 0)
                  / max(1, sum(1 for e in evals if e.get("faithfulness_score", -1) >= 0)), 3)
        )
        out[model] = {
            "total": total,
            "faithful": faithful,
            "issues_found": total - faithful,
            "issue_rate_pct": _pct(total - faithful, total),
            "avg_faithfulness_score": avg_score,
            "issue_type_counts": dict(issue_counts),
        }
    return out


def compute_detection_performance(results: Dict) -> Dict:
    """Per-model detection method statistics (precision/recall computed if labels exist)."""
    out = {}
    for model, data in results.items():
        out[model] = {}
        anns = {(a["interview_id"], a["query_type"]): a.get("overall_label")
                for a in data["annotations"] if a.get("interview_id")}

        for method_key in ["selfcheck", "minicheck", "alignscore", "ragas"]:
            rows = data[method_key]
            if not rows:
                out[model][method_key] = {"n": 0}
                continue

            scores = [r.get("score", r.get("faithfulness_score", -1)) for r in rows]
            scores = [s for s in scores if s is not None and s >= 0]

            # Try to compute binary predictions vs RQ1 ground truth
            tp = fp = tn = fn = 0
            for r in rows:
                key = (r.get("interview_id"), r.get("query_type"))
                gt  = anns.get(key)
                pred_score = r.get("score", r.get("faithfulness_score", -1))
                if gt is None or pred_score < 0:
                    continue
                # low faithfulness score → predicted hallucinated
                threshold = 0.5
                pred_hall = pred_score < threshold
                gt_hall   = gt == "HALLUCINATED"
                if pred_hall and gt_hall:     tp += 1
                elif pred_hall and not gt_hall: fp += 1
                elif not pred_hall and gt_hall: fn += 1
                else:                           tn += 1

            precision = round(tp / (tp + fp), 3) if (tp + fp) else 0.0
            recall    = round(tp / (tp + fn), 3) if (tp + fn) else 0.0
            f1        = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0
            accuracy  = round((tp + tn) / (tp + fp + tn + fn), 3) if (tp + fp + tn + fn) else 0.0

            out[model][method_key] = {
                "n": len(rows),
                "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
    return out


def build_comparison_json(results: Dict, results_base: str) -> Dict:
    """Build the complete comparison dict and save to JSON."""
    print("\nComputing hallucination rates...")
    hal_rates = compute_hallucination_rates(results)
    print("Computing DiaHaLu rates...")
    dial_rates = compute_diahalu_rates(results)
    print("Computing detection performance...")
    det_perf   = compute_detection_performance(results)

    comparison = {
        "models_compared": list(results.keys()),
        "hallucination_rates": hal_rates,
        "diahalu_rates":       dial_rates,
        "detection_performance": det_perf,
    }

    out_dir = Path(results_base) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison JSON → {out_path}")
    return comparison


# ── PDF generation ────────────────────────────────────────────────────────────

def _safe_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                         Paragraph, Spacer, PageBreak, HRFlowable)
        return True
    except ImportError:
        return False


def _make_table(data: List[List], col_widths=None, header_color=None):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    header_color = header_color or colors.HexColor("#2c3e50")
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


def generate_pdf(comparison: Dict, output_path: str):
    """Generate the comparison PDF report using ReportLab."""
    if not _safe_reportlab():
        print("  reportlab not installed — skipping PDF. Run: pip install reportlab")
        return

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer, PageBreak, HRFlowable)

    PAGE = landscape(A4)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=PAGE,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18,
                         textColor=colors.HexColor("#2c3e50"), spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                         textColor=colors.HexColor("#2980b9"), spaceAfter=4)
    BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9, spaceAfter=4)

    story = []
    models = comparison.get("models_compared", MODELS)
    hal    = comparison.get("hallucination_rates", {})
    dial   = comparison.get("diahalu_rates", {})
    det    = comparison.get("detection_performance", {})

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 3*cm),
        Paragraph("Hallucination in Interview-Based RAG Systems", H1),
        Paragraph("Cross-Model Comparison Report", H2),
        Paragraph(
            "GPT-4o-mini  ·  Qwen2.5-7B (Together.ai)  ·  Mistral-7B (Together.ai)",
            BODY),
        Spacer(1, 0.5*cm),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2980b9")),
        Spacer(1, 0.5*cm),
        Paragraph(
            "RQ1: What types of hallucinations do LLMs produce when answering "
            "questions about interview transcripts? (RAGTruth · Huang · DiaHaLu)",
            BODY),
        Paragraph(
            "RQ2: Which automatic detection method best identifies these "
            "hallucinations? (SelfCheckGPT · MiniCheck · AlignScore)",
            BODY),
        PageBreak(),
    ]

    # ── Table 1: Overall hallucination rates ────────────────────────────────
    story.append(Paragraph("Table 1 — Overall Hallucination Rates (RQ1, RAGTruth)", H2))
    header = ["Model", "Total\nResponses", "Hallucinated", "Faithful",
              "Hallucination\nRate (%)", "Avg Faithfulness\nScore"]
    rows   = [header]
    for m in models:
        d = hal.get(m, {})
        rows.append([
            m,
            str(d.get("total", "—")),
            str(d.get("hallucinated", "—")),
            str(d.get("faithful", "—")),
            f"{d.get('hallucination_rate_pct', '—')}%",
            str(d.get("avg_faithfulness_score", "—")),
        ])
    story.append(_make_table(rows, col_widths=[4*cm, 3*cm, 3*cm, 3*cm, 3.5*cm, 3.5*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Table 2: Hallucination types ────────────────────────────────────────
    story.append(Paragraph("Table 2 — Hallucination Type Breakdown (RQ1)", H2))
    header2 = ["Hallucination Type"] + models
    rows2 = [header2]
    for ht in HAL_TYPES:
        row = [ht]
        for m in models:
            count = hal.get(m, {}).get("type_counts", {}).get(ht, 0)
            row.append(str(count))
        rows2.append(row)
    col_w2 = [6*cm] + [3*cm] * len(models)
    story.append(_make_table(rows2, col_widths=col_w2))
    story.append(Spacer(1, 0.5*cm))

    # ── Table 3: Query-type breakdown ───────────────────────────────────────
    story.append(Paragraph("Table 3 — Hallucination Rate by Query Type (%)", H2))
    header3 = ["Query Type"] + [f"{m}\n(%)" for m in models]
    rows3 = [header3]
    for qt in QUERY_TYPES:
        row = [qt]
        for m in models:
            rate = hal.get(m, {}).get("by_query_type", {}).get(qt, {}).get("rate_pct", "—")
            row.append(f"{rate}%" if rate != "—" else "—")
        rows3.append(row)
    col_w3 = [5*cm] + [3.5*cm] * len(models)
    story.append(_make_table(rows3, col_widths=col_w3))
    story.append(PageBreak())

    # ── Table 4: Language breakdown ─────────────────────────────────────────
    story.append(Paragraph("Table 4 — Hallucination Rate by Language", H2))
    header4 = ["Language", "Metric"] + models
    rows4 = [header4]
    for lang in ["en", "nl"]:
        for metric, key in [("Total", "total"), ("Hallucinated", "hallucinated"),
                             ("Rate (%)", "rate_pct")]:
            row = [lang if metric == "Total" else "", metric]
            for m in models:
                val = hal.get(m, {}).get("by_language", {}).get(lang, {}).get(key, "—")
                row.append(f"{val}%" if key == "rate_pct" and val != "—" else str(val))
            rows4.append(row)
    col_w4 = [2*cm, 3.5*cm] + [3*cm] * len(models)
    story.append(_make_table(rows4, col_widths=col_w4))
    story.append(Spacer(1, 0.5*cm))

    # ── Table 5: DiaHaLu dialogue issues ────────────────────────────────────
    story.append(Paragraph("Table 5 — DiaHaLu Dialogue-Level Issues (RQ1, Exp 3)", H2))
    header5 = ["Dialogue Issue Type"] + models
    rows5 = [header5]
    # Summary row
    for label, key in [("Total evaluated", "total"), ("With issues", "issues_found"),
                        ("Issue rate (%)", "issue_rate_pct"),
                        ("Avg faithfulness score", "avg_faithfulness_score")]:
        row = [label]
        for m in models:
            val = dial.get(m, {}).get(key, "—")
            row.append(f"{val}%" if key == "issue_rate_pct" and val != "—" else str(val))
        rows5.append(row)
    # Per-type counts
    for dt in DIAL_TYPES:
        row = [dt]
        for m in models:
            count = dial.get(m, {}).get("issue_type_counts", {}).get(dt, 0)
            row.append(str(count))
        rows5.append(row)
    col_w5 = [6*cm] + [3*cm] * len(models)
    story.append(_make_table(rows5, col_widths=col_w5))
    story.append(PageBreak())

    # ── Table 6: Detection performance ──────────────────────────────────────
    story.append(Paragraph("Table 6 — Detection Method Performance (RQ2)", H2))
    story.append(Paragraph(
        "Predictions binarised at faithfulness threshold = 0.5 against RQ1 ground truth labels.",
        BODY))
    story.append(Spacer(1, 0.2*cm))

    for method in ["selfcheck", "minicheck", "alignscore", "ragas"]:
        story.append(Paragraph(f"  {method.upper()}", ParagraphStyle(
            "sub", parent=BODY, fontName="Helvetica-Bold", fontSize=10)))
        header6 = ["Model", "N", "Avg Score", "Precision", "Recall", "F1", "Accuracy"]
        rows6 = [header6]
        for m in models:
            d = det.get(m, {}).get(method, {})
            if not d or d.get("n", 0) == 0:
                rows6.append([m, "—", "—", "—", "—", "—", "—"])
            else:
                rows6.append([
                    m,
                    str(d.get("n", "—")),
                    str(d.get("avg_score", "—")),
                    str(d.get("precision", "—")),
                    str(d.get("recall",    "—")),
                    str(d.get("f1",        "—")),
                    str(d.get("accuracy",  "—")),
                ])
        col_w6 = [4*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
        story.append(_make_table(rows6, col_widths=col_w6,
                                  header_color=colors.HexColor("#1a5276")))
        story.append(Spacer(1, 0.4*cm))

    doc.build(story)
    print(f"Saved PDF → {output_path}")


# ── main entry point ─────────────────────────────────────────────────────────

def generate_comparison_report(results_base: str = "results",
                                output_dir: str = "results/comparison"):
    """Load all model results, compute stats, save JSON + PDF."""
    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON REPORT")
    print("=" * 60)
    print(f"Results base : {results_base}")
    print(f"Output dir   : {output_dir}")

    results    = load_model_results(results_base)
    if not results:
        print("No model results found — run pipeline steps first.")
        return

    comparison = build_comparison_json(results, results_base)

    pdf_path = Path(output_dir) / "comparison_report.pdf"
    generate_pdf(comparison, pdf_path)

    print("\nComparison complete.")
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default="results")
    args = parser.parse_args()
    generate_comparison_report(
        results_base=args.results_base,
        output_dir=str(Path(args.results_base) / "comparison"),
    )
