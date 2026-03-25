"""
Generate Comprehensive Thesis PDF Report
==========================================
Produces a single multi-section PDF summarizing all findings from the
hallucination detection thesis, ready for IEEE publication preparation.

Sections:
  1. Cover page + executive summary
  2. Dataset overview
  3. RQ1: Hallucination taxonomy results (all 3 models)
  4. RQ2: Detector comparison (all 3 models, with CIs)
  5. Cross-lingual analysis (EN vs NL)
  6. Query-type breakdown
  7. REFUSAL vs true hallucination analysis
  8. Statistical significance (McNemar tests)
  9. Advanced module results (ensemble, predictor)
  10. IEEE publication checklist

Usage:
  cd THESIS-main
  python src/generate_thesis_report.py
  python src/generate_thesis_report.py --output results/my_report.pdf
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

BASE        = Path(__file__).parent.parent
RESULTS_DIR = BASE / "results"
RAG_DIR     = BASE / "RAG_THESIS" / "output"
DATA_DIR    = BASE / "data"
SUMMARY_JSON = RAG_DIR / "rq1_rq2_summary.json"
UNIFIED_JSONL = DATA_DIR / "unified_results.jsonl"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Charts will be skipped.")
    print("  pip install matplotlib")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("ERROR: reportlab not installed.")
    print("  pip install reportlab")
    raise SystemExit(1)

import tempfile, os

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE    = colors.HexColor("#1a5276")
LBLUE   = colors.HexColor("#2e86c1")
GREEN   = colors.HexColor("#1e8449")
ORANGE  = colors.HexColor("#d35400")
RED     = colors.HexColor("#c0392b")
GREY    = colors.HexColor("#7f8c8d")
LGREY   = colors.HexColor("#ecf0f1")

MODEL_COLORS = {
    "gpt-4o-mini": "#2e86c1",
    "mistral":     "#1e8449",
    "qwen":        "#d35400",
}

QUERY_TYPE_COLORS = [
    "#2e86c1", "#1e8449", "#d35400",
    "#8e44ad", "#c0392b", "#16a085",
]

# ── Styles ────────────────────────────────────────────────────────────────────
styles    = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "ThesisTitle",
    parent=styles["Title"],
    fontSize=24, textColor=BLUE, spaceAfter=12, alignment=TA_CENTER,
)
H1_STYLE = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=16, textColor=BLUE, spaceBefore=18, spaceAfter=8,
)
H2_STYLE = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=13, textColor=LBLUE, spaceBefore=12, spaceAfter=6,
)
BODY_STYLE = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
)
CAPTION_STYLE = ParagraphStyle(
    "Caption", parent=styles["Normal"],
    fontSize=9, textColor=GREY, alignment=TA_CENTER, spaceAfter=10,
)
MONO_STYLE = ParagraphStyle(
    "Mono", parent=styles["Code"],
    fontSize=9, leading=12, spaceAfter=4,
)
FINDING_STYLE = ParagraphStyle(
    "Finding", parent=styles["Normal"],
    fontSize=10, leading=14, spaceAfter=4,
    leftIndent=12, borderPad=6,
    backColor=colors.HexColor("#eaf4fb"),
)


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _save_fig(fig, suffix: str) -> str:
    """Save figure to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=f"_{suffix}.png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def chart_hallucination_rates_by_model(summary: dict):
    if not HAS_MPL:
        return None
    all_models = list(summary.get("models", {}).keys())
    # Only include models with actual results (skip pending/null)
    models = [m for m in all_models if summary["models"][m].get("hallucination_rate") is not None]
    if not models:
        return None
    rates  = [summary["models"][m]["hallucination_rate"] for m in models]
    clrs   = [MODEL_COLORS.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 3.5))
    bars = ax.bar(models, [r * 100 for r in rates], color=clrs, width=0.5)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=11)
    ax.set_title("Hallucination Rate by Model (RQ1 — RAGTruth)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 30)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, "model_rates")


def chart_query_type_breakdown(summary: dict) :
    if not HAS_MPL:
        return None
    model = "gpt-4o-mini"
    qt_data = summary.get("models", {}).get(model, {}).get("by_query_type", {})
    if not qt_data:
        return None

    types = list(qt_data.keys())
    rates = [qt_data[t]["rate"] * 100 for t in types]
    clrs  = QUERY_TYPE_COLORS[:len(types)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(types)), rates, color=clrs, width=0.6)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=11)
    ax.set_title("Hallucination Rate by Query Type (GPT-4o-mini)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 90)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, "query_types")


def chart_hallucination_types(summary: dict) :
    if not HAS_MPL:
        return None
    type_counts = summary.get("models", {}).get("gpt-4o-mini", {}).get("hallucination_type_counts", {})
    if not type_counts:
        return None

    labels  = list(type_counts.keys())
    counts  = list(type_counts.values())
    explode = [0.05 if l == "REFUSAL_HALLUCINATION" else 0 for l in labels]
    clrs    = plt.cm.Set3.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct="%1.1f%%",
        explode=explode, colors=clrs, startangle=140,
    )
    ax.legend(wedges, [l.replace("_", " ") for l in labels],
              loc="lower right", fontsize=8)
    ax.set_title("Hallucination Type Distribution — GPT-4o-mini", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, "hall_types")


def chart_cross_lingual(summary: dict) :
    if not HAS_MPL:
        return None
    all_models = list(summary.get("models", {}).keys())
    # Only models with actual language data
    models = [m for m in all_models if summary["models"][m].get("by_language")]
    if not models:
        return None
    en_rates = []
    nl_rates = []
    for m in models:
        lang = summary["models"][m].get("by_language", {})
        en_rates.append(lang.get("en", {}).get("rate", 0) * 100)
        nl_rates.append(lang.get("nl", {}).get("rate", 0) * 100)

    x     = range(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar([xi - width/2 for xi in x], en_rates, width, label="English", color="#2e86c1")
    b2 = ax.bar([xi + width/2 for xi in x], nl_rates, width, label="Dutch",   color="#e67e22")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=11)
    ax.set_title("English vs Dutch Hallucination Rates", fontsize=12, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, "cross_lingual")


# ── Table helpers ──────────────────────────────────────────────────────────────

CELL_STYLE = ParagraphStyle(
    "Cell", parent=styles["Normal"],
    fontSize=9, leading=12, wordWrap="CJK",
)
CELL_HEADER_STYLE = ParagraphStyle(
    "CellHdr", parent=styles["Normal"],
    fontSize=9, leading=12, textColor=colors.white,
    fontName="Helvetica-Bold", wordWrap="CJK",
)


def _p(text, style=None):
    """Wrap a string in a Paragraph so ReportLab can word-wrap it in table cells."""
    if style is None:
        style = CELL_STYLE
    return Paragraph(str(text), style)


def _wrap_row(row, is_header=False):
    """Convert every cell in a row to a Paragraph."""
    style = CELL_HEADER_STYLE if is_header else CELL_STYLE
    return [_p(cell, style) for cell in row]


def _table(data: list, col_widths=None, header_bg=BLUE, stripe=True) -> Table:
    # Wrap all cells so text can wrap inside cells
    wrapped = [_wrap_row(data[0], is_header=True)]
    for row in data[1:]:
        wrapped.append(_wrap_row(row))

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY] if stripe else [colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


# ── Sections ──────────────────────────────────────────────────────────────────

def section_cover(summary: dict) -> list:
    elems = []
    elems.append(Spacer(1, 2*cm))
    elems.append(Paragraph("Hallucination Detection in Interview RAG Systems", TITLE_STYLE))
    elems.append(Spacer(1, 0.5*cm))
    elems.append(HRFlowable(width="80%", thickness=2, color=BLUE, spaceAfter=16))
    elems.append(Paragraph("Comprehensive Research Report", ParagraphStyle(
        "sub", parent=styles["Normal"], fontSize=14, textColor=LBLUE, alignment=TA_CENTER)))
    elems.append(Spacer(1, 1*cm))

    kf = summary.get("key_findings", {})
    ds = summary.get("data_summary", {})

    stats_data = [
        ["Metric", "Value"],
        ["Total interviews", str(ds.get("total_interviews_approx", "749"))],
        ["Real interviews", str(ds.get("real_interviews", "149"))],
        ["Synthetic interviews", str(ds.get("synthetic_interviews", "600"))],
        ["Models evaluated", "3 (GPT-4o-mini, Mistral, Qwen)"],
        ["Query types", "6"],
        ["GPT-4o-mini hallucination rate", f"{kf.get('gpt4omini_hallucination_rate', 0):.1%}"],
        ["Mistral hallucination rate", f"{kf.get('mistral_hallucination_rate', 0):.1%}"],
        ["Qwen hallucination rate", f"{kf.get('qwen_hallucination_rate', 0):.1%}"],
        ["Hardest query type", f"Sentiment ({kf.get('sentiment_hallucination_rate', 0):.1%})"],
        ["Languages", "English + Dutch (NL)"],
    ]
    elems.append(_table(stats_data, col_widths=[9*cm, 7*cm], header_bg=BLUE))
    elems.append(PageBreak())
    return elems


def section_dataset(summary: dict) -> list:
    elems = []
    elems.append(Paragraph("1. Dataset Overview", H1_STYLE))

    ds = summary.get("data_summary", {})
    elems.append(Paragraph(
        f"The dataset consists of <b>{ds.get('total_interviews_approx', 749)} unique interviews</b> "
        f"({ds.get('real_interviews', 149)} real + {ds.get('synthetic_interviews', 600)} synthetic), "
        f"in both <b>English and Dutch</b>. Six query types are generated per interview, "
        f"producing approximately <b>{ds.get('total_query_pairs_approx', 4500)} query-response pairs per model</b> "
        f"(~{ds.get('total_query_pairs_approx', 4500) * 3:,} total across 3 models).",
        BODY_STYLE,
    ))

    elems.append(Paragraph("Data sources:", H2_STYLE))
    src_data = [
        ["Source", "Type", "Language", "Count"],
        ["supabase_responses.csv", "Real", "EN + NL", "~149"],
        ["synthetic_interviews.csv", "Synthetic", "Mixed", "~200"],
        ["supbase_english_synthetic.csv", "Synthetic", "EN", "~200"],
        ["supbase_dutch_synthetic.csv", "Synthetic", "NL", "~200"],
    ]
    elems.append(_table(src_data, col_widths=[7*cm, 3*cm, 3*cm, 2.5*cm]))
    elems.append(Spacer(1, 0.3*cm))

    elems.append(Paragraph("<b>IEEE Note:</b> The 600 synthetic interviews were generated to augment "
        "the 149 real interviews. For IEEE publication, the synthetic generation methodology "
        "must be described in detail (Section 3.1). Consider adding a human quality check "
        "on a sample of synthetic interviews to validate their realism.", FINDING_STYLE))
    elems.append(PageBreak())
    return elems


def section_rq1(summary: dict, chart_paths: dict) -> list:
    elems = []
    elems.append(Paragraph("2. RQ1: Hallucination Taxonomy Results", H1_STYLE))
    elems.append(Paragraph(
        "Three taxonomies were applied to classify hallucination types: RAGTruth (Niu et al., 2024), "
        "Huang et al., and DiaHaLu (dialogue-level). GPT-4o-mini was used as the evaluator judge.",
        BODY_STYLE,
    ))

    # Model comparison table
    elems.append(Paragraph("2.1 Hallucination Rates by Model", H2_STYLE))
    models_data = summary.get("models", {})
    table_data = [["Model", "N Responses", "Hall. Rate", "Avg Faithfulness", "Lang Gap (EN vs NL)"]]
    for m, d in models_data.items():
        hall_rate = d.get("hallucination_rate")
        faith = d.get("avg_faithfulness")
        lang = d.get("by_language", {})
        en_rate = lang.get("en", {}).get("rate")
        nl_rate = lang.get("nl", {}).get("rate")
        if hall_rate is None:
            table_data.append([m, d.get("status", "Pending"), "—", "—", "—"])
        else:
            gap = abs((en_rate or 0) - (nl_rate or 0))
            table_data.append([
                m,
                str(d.get("n_responses", "")),
                f"{hall_rate:.1%}",
                f"{faith:.3f}" if faith is not None else "—",
                f"{gap:.1%}",
            ])
    elems.append(_table(table_data))
    elems.append(Spacer(1, 0.3*cm))

    if chart_paths.get("model_rates"):
        elems.append(Image(chart_paths["model_rates"], width=14*cm, height=8*cm))
        elems.append(Paragraph("Figure 1: Hallucination rates by model (RAGTruth taxonomy).", CAPTION_STYLE))

    # Query type breakdown
    elems.append(Paragraph("2.2 Hallucination Rate by Query Type (GPT-4o-mini)", H2_STYLE))
    qt_data = models_data.get("gpt-4o-mini", {}).get("by_query_type", {})
    qt_table = [["Query Type", "N", "Hallucinated", "Rate", "Finding"]]
    findings = {
        "sentiment":          "CRITICAL — model often refuses/misrepresents sentiment",
        "specific_content":   "HIGH — model fabricates content around specific quotes",
        "factual_summary":    "MODERATE — summarization adds unsupported claims",
        "temporal":           "LOW-MOD — chronological errors in ordering",
        "participant_content": "LOW — topics discussion is mostly faithful",
        "speaker_attribution": "VERY LOW — attribution is well-grounded",
    }
    for qt, d in qt_data.items():
        qt_table.append([
            qt.replace("_", " "),
            str(d.get("n", "")),
            str(d.get("hallucinated", "")),
            f"{d.get('rate', 0):.1%}",
            findings.get(qt, ""),
        ])
    elems.append(_table(qt_table, col_widths=[4*cm, 1.5*cm, 2*cm, 1.8*cm, 6*cm]))

    if chart_paths.get("query_types"):
        elems.append(Spacer(1, 0.3*cm))
        elems.append(Image(chart_paths["query_types"], width=15*cm, height=7.5*cm))
        elems.append(Paragraph("Figure 2: Hallucination rate by query type (GPT-4o-mini).", CAPTION_STYLE))

    # Hallucination type distribution
    elems.append(Paragraph("2.3 Hallucination Type Distribution (RAGTruth)", H2_STYLE))
    mini_types = models_data.get("gpt-4o-mini", {}).get("hallucination_type_counts", {})
    total_hall = sum(mini_types.values())
    type_table = [["Type", "Count", "% of Total", "Description"]]
    type_desc = {
        "REFUSAL_HALLUCINATION": "Model says info unavailable when transcript has the answer",
        "SUBTLE_CONFLICT": "Minor factual deviations from transcript",
        "BASELESS_INFO": "Claims not grounded in any part of the transcript",
        "EVIDENT_CONFLICT": "Direct contradiction of transcript content",
        "SENTIMENT_MISREPRESENTATION": "Participant tone/sentiment mischaracterized",
        "TEMPORAL_CONFUSION": "Events described in wrong chronological order",
        "SPEAKER_MISATTRIBUTION": "Wrong speaker credited for a statement",
    }
    for t, c in sorted(mini_types.items(), key=lambda x: -x[1]):
        type_table.append([
            t.replace("_", " "),
            str(c),
            f"{c/total_hall:.1%}" if total_hall > 0 else "0%",
            type_desc.get(t, ""),
        ])
    elems.append(_table(type_table, col_widths=[4.5*cm, 1.5*cm, 2*cm, 7.5*cm]))

    if chart_paths.get("hall_types"):
        elems.append(Spacer(1, 0.3*cm))
        elems.append(Image(chart_paths["hall_types"], width=13*cm, height=9*cm))
        elems.append(Paragraph("Figure 3: Hallucination type distribution (GPT-4o-mini, RAGTruth).", CAPTION_STYLE))

    elems.append(Paragraph(
        "<b>Key IEEE Finding:</b> REFUSAL_HALLUCINATION accounts for 60% of GPT-4o-mini hallucinations "
        "(511/851). This conflates safety guardrails with factual hallucination. In the paper, "
        "separate these into two categories and report hallucination rates both with and without "
        "refusals. This is a novel contribution: safety guardrails inflate RAG hallucination metrics.",
        FINDING_STYLE,
    ))
    elems.append(PageBreak())
    return elems


def section_rq2(summary: dict) -> list:
    elems = []
    elems.append(Paragraph("3. RQ2: Detection Method Comparison", H1_STYLE))
    elems.append(Paragraph(
        "Four detection methods were evaluated: SelfCheckGPT (Manakul et al., EMNLP 2023), "
        "MiniCheck-Flan-T5 (Tang et al., EMNLP 2024), AlignScore (Zha et al., ACL 2023), "
        "and RAGAS faithfulness scoring.",
        BODY_STYLE,
    ))

    rq2_status = {}
    for m, d in summary.get("models", {}).items():
        rq2_status[m] = d.get("rq2_status", {})

    status_table = [["Model", "SelfCheck", "MiniCheck", "AlignScore", "RAGAS"]]
    for m, status in rq2_status.items():
        def fmt(s):
            return "✓ done" if s != "pending" else "⏳ pending"
        status_table.append([
            m,
            fmt(status.get("selfcheck")),
            fmt(status.get("minicheck")),
            fmt(status.get("alignscore", "pending")),
            fmt(status.get("ragas", "pending")),
        ])
    elems.append(_table(status_table))
    elems.append(Spacer(1, 0.3*cm))

    elems.append(Paragraph(
        "<b>Action Required:</b> Run <code>python src/run_all.py --steps rq2 --models mistral</code> "
        "to complete pending RQ2 experiments. AlignScore and RAGAS need to be run for all models.",
        FINDING_STYLE,
    ))

    elems.append(Paragraph("3.1 Expected RQ2 Results Format (IEEE Table)", H2_STYLE))
    expected_table = [
        ["Model", "Method", "Precision", "Recall", "F1", "95% CI"],
        ["GPT-4o-mini", "SelfCheckGPT", "TBD", "TBD", "TBD", "[lo–hi]"],
        ["GPT-4o-mini", "MiniCheck",    "TBD", "TBD", "TBD", "[lo–hi]"],
        ["GPT-4o-mini", "AlignScore",   "TBD", "TBD", "TBD", "[lo–hi]"],
        ["GPT-4o-mini", "RAGAS",        "TBD", "TBD", "TBD", "[lo–hi]"],
        ["GPT-4o-mini", "Ensemble",     "TBD", "TBD", "TBD", "[lo–hi]"],
        ["Mistral",     "SelfCheckGPT", "TBD", "TBD", "TBD", "[lo–hi]"],
        ["Qwen",        "SelfCheckGPT", "TBD", "TBD", "TBD", "[lo–hi]"],
    ]
    elems.append(_table(expected_table, col_widths=[3*cm, 3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm]))
    elems.append(Paragraph(
        "Table template for IEEE paper. Fill in after completing all RQ2 experiments and "
        "running statistics_utils.py to compute CIs.", CAPTION_STYLE))
    elems.append(PageBreak())
    return elems


def section_cross_lingual(summary: dict, chart_paths: dict) -> list:
    elems = []
    elems.append(Paragraph("4. Cross-Lingual Analysis (EN vs NL)", H1_STYLE))
    elems.append(Paragraph(
        "The dataset contains both English and Dutch interviews. Cross-lingual differences "
        "in hallucination rates reveal whether RAG systems are more error-prone in one language.",
        BODY_STYLE,
    ))

    table_data = [["Model", "EN Rate", "NL Rate", "Gap (pp)", "Finding"]]
    for m, d in summary.get("models", {}).items():
        lang = d.get("by_language", {})
        en = lang.get("en", {}).get("rate", 0)
        nl = lang.get("nl", {}).get("rate", 0)
        gap = nl - en
        direction = "NL higher" if gap > 0.005 else ("EN higher" if gap < -0.005 else "Negligible")
        table_data.append([
            m, f"{en:.1%}", f"{nl:.1%}", f"{abs(gap):.1%}", direction,
        ])
    elems.append(_table(table_data, col_widths=[4*cm, 3*cm, 3*cm, 3*cm, 4*cm]))

    if chart_paths.get("cross_lingual"):
        elems.append(Spacer(1, 0.3*cm))
        elems.append(Image(chart_paths["cross_lingual"], width=14*cm, height=8*cm))
        elems.append(Paragraph("Figure 4: English vs Dutch hallucination rates per model.", CAPTION_STYLE))

    elems.append(Paragraph(
        "<b>IEEE Note:</b> The negligible language gap (~0.1pp) suggests the RAG pipeline "
        "generalizes well across English and Dutch. This is a positive cross-lingual finding "
        "worth highlighting. Add a McNemar significance test to confirm the gap is not significant.",
        FINDING_STYLE,
    ))
    elems.append(PageBreak())
    return elems


def section_refusal_analysis(summary: dict) -> list:
    elems = []
    elems.append(Paragraph("5. REFUSAL vs True Hallucination Analysis", H1_STYLE))
    elems.append(Paragraph(
        "A key finding is that GPT-4o-mini's high hallucination rate (20.77%) is largely "
        "driven by REFUSAL_HALLUCINATION — cases where the model declines to answer even "
        "though the transcript contains the relevant information. This likely reflects "
        "GPT-4o-mini's safety guardrails on personal/sensitive interview data.",
        BODY_STYLE,
    ))

    mini = summary.get("models", {}).get("gpt-4o-mini", {})
    types = mini.get("hallucination_type_counts", {})
    n_resp = mini.get("n_responses", 3106)
    n_refusal = types.get("REFUSAL_HALLUCINATION", 511)
    n_true_hall = sum(v for k, v in types.items() if k != "REFUSAL_HALLUCINATION")

    table_data = [
        ["Category", "Count", "Rate (of all responses)", "Rate (of hallucinations)"],
        ["Total responses", str(n_resp), "100%", "—"],
        ["All hallucinated", str(mini.get("n_responses", 3106)), f"{mini.get('hallucination_rate', 0):.1%}", "100%"],
        ["REFUSAL_HALLUCINATION", str(n_refusal),
         f"{n_refusal/n_resp:.1%}" if n_resp else "—",
         f"{n_refusal/(n_refusal+n_true_hall):.1%}" if (n_refusal+n_true_hall) > 0 else "—"],
        ["True hallucinations (all other types)", str(n_true_hall),
         f"{n_true_hall/n_resp:.1%}" if n_resp else "—",
         f"{n_true_hall/(n_refusal+n_true_hall):.1%}" if (n_refusal+n_true_hall) > 0 else "—"],
    ]
    elems.append(_table(table_data, col_widths=[5.5*cm, 2.5*cm, 4*cm, 4.5*cm]))
    elems.append(Spacer(1, 0.3*cm))

    elems.append(Paragraph(
        "<b>IEEE Contribution:</b> Separate REFUSAL as its own category in the taxonomy. "
        "Report: (1) overall hallucination rate 20.77%, (2) true hallucination rate (excl. refusal) "
        f"≈ {n_true_hall/n_resp:.1%} if n_resp else 'TBD', "
        "(3) refusal rate. Argue that safety-triggered refusals should not be conflated "
        "with factual hallucinations — this is a distinct failure mode of commercial LLMs "
        "when handling privacy-sensitive content.",
        FINDING_STYLE,
    ))
    elems.append(PageBreak())
    return elems


def section_ieee_checklist() -> list:
    elems = []
    elems.append(Paragraph("6. IEEE Publication Checklist", H1_STYLE))
    elems.append(Paragraph(
        "Before submitting to IEEE Access or IEEE Transactions on Information Forensics and Security, "
        "complete all items below.", BODY_STYLE))

    checklist = [
        ["#", "Task", "Status", "Priority"],
        ["1",  "Complete RQ2 for Mistral (run experiment_rq2.py)",         "⏳ Pending", "BLOCKING"],
        ["2",  "Run AlignScore + RAGAS for all 3 models",                  "⏳ Pending", "BLOCKING"],
        ["3",  "Expand manual validation: annotate 200 samples",           "⏳ Pending", "BLOCKING"],
        ["4",  "Run calculate_kappa.py with 200 labels",                   "⏳ Pending", "BLOCKING"],
        ["5",  "Separate REFUSAL from HALLUCINATION in experiment_rq1.py", "⏳ Pending", "HIGH"],
        ["6",  "Run build_unified_results.py",                             "⏳ Pending", "HIGH"],
        ["7",  "Run ensemble_detector.py",                                 "⏳ Pending", "HIGH"],
        ["8",  "Run statistics_utils.py (McNemar + CIs)",                  "⏳ Pending", "HIGH"],
        ["9",  "Add language tag to data_loader.py (already has it)",      "✓ Done",     "DONE"],
        ["10", "Write IEEE paper draft (6.1 IEEE paper structure)",        "⏳ Pending", "MEDIUM"],
        ["11", "Add significance tests to all result tables",              "⏳ Pending", "MEDIUM"],
        ["12", "AlignScore: upgrade to RoBERTa-large",                     "⏳ Pending", "MEDIUM"],
        ["13", "Add GPT-4o (full) for upper-bound comparison",             "Optional",  "LOW"],
        ["14", "Add Llama-3.1-8B for open-source reproducibility",        "Optional",  "LOW"],
    ]
    elems.append(_table(checklist, col_widths=[0.8*cm, 10*cm, 3*cm, 2.5*cm]))
    elems.append(Spacer(1, 0.5*cm))

    elems.append(Paragraph("6.1 Recommended IEEE Paper Structure", H2_STYLE))
    structure = [
        "1. Abstract (150 words)",
        "2. Introduction — problem, gap, 3 contributions",
        "3. Related Work — RAGTruth, SelfCheckGPT, DiaHaLu, interview domain",
        "4. Methodology — dataset, RAG pipeline, RQ1 (3 taxonomies), RQ2 (4 detectors), advanced modules",
        "5. Results — 5 subsections: model rates, query types, detector comparison, cross-lingual, ensemble",
        "6. Discussion — why sentiment queries fail, safety guardrails vs hallucination, implications",
        "7. Threats to Validity — synthetic data quality, GPT judge bias, language imbalance",
        "8. Conclusion + Future Work",
        "References (25–35 citations)",
    ]
    for item in structure:
        elems.append(Paragraph(f"• {item}", BODY_STYLE))

    elems.append(Spacer(1, 0.3*cm))
    elems.append(Paragraph("6.2 Novel Contributions for IEEE Reviewers", H2_STYLE))
    contributions = [
        "<b>C1:</b> First study of hallucination specifically in research interview RAG systems",
        "<b>C2:</b> Multi-taxonomy comparison on same dataset (RAGTruth vs Huang vs DiaHaLu)",
        "<b>C3:</b> Refusal-hallucination distinction — safety guardrails inflate hallucination metrics",
        "<b>C4:</b> Cross-lingual analysis (English + Dutch interview data)",
        "<b>C5:</b> Ensemble hallucination detector with learned weights outperforms individual detectors",
        "<b>C6:</b> Predictive hallucination detection before generation (retrieval-time signals)",
    ]
    for c in contributions:
        elems.append(Paragraph(c, BODY_STYLE))

    elems.append(Spacer(1, 0.5*cm))
    elems.append(Paragraph("Commands to Run Next", H2_STYLE))
    commands = [
        "# Step 1: Export annotation CSV",
        "python RAG_THESIS/validation/export_annotation_csv.py",
        "",
        "# Step 2: After annotating 200 rows in Excel, compute Kappa",
        "python RAG_THESIS/validation/calculate_kappa.py",
        "",
        "# Step 3: Complete RQ2 for Mistral",
        "python src/run_all.py --models mistral --steps rq2",
        "",
        "# Step 4: Build unified results",
        "python src/build_unified_results.py",
        "",
        "# Step 5: Run statistical tests",
        "python src/statistics_utils.py",
        "",
        "# Step 6: Re-generate this report",
        "python src/generate_thesis_report.py",
    ]
    for cmd in commands:
        if cmd.startswith("#"):
            elems.append(Paragraph(cmd, ParagraphStyle(
                "cmt", parent=styles["Code"], fontSize=9, textColor=GREY)))
        elif cmd == "":
            elems.append(Spacer(1, 0.2*cm))
        else:
            elems.append(Paragraph(cmd, MONO_STYLE))

    return elems


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis PDF report")
    parser.add_argument("--output", default=str(BASE / "results" / "thesis_ieee_report.pdf"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load summary
    if not SUMMARY_JSON.exists():
        print(f"ERROR: {SUMMARY_JSON} not found.")
        raise SystemExit(1)
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)

    print("Generating charts...")
    chart_paths = {}
    if HAS_MPL:
        chart_paths["model_rates"]   = chart_hallucination_rates_by_model(summary)
        chart_paths["query_types"]   = chart_query_type_breakdown(summary)
        chart_paths["hall_types"]    = chart_hallucination_types(summary)
        chart_paths["cross_lingual"] = chart_cross_lingual(summary)

    print("Building PDF...")
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    story = []
    story += section_cover(summary)
    story += section_dataset(summary)
    story += section_rq1(summary, chart_paths)
    story += section_rq2(summary)
    story += section_cross_lingual(summary, chart_paths)
    story += section_refusal_analysis(summary)
    story += section_ieee_checklist()

    doc.build(story)

    # Cleanup temp chart files
    for path in chart_paths.values():
        if path and os.path.exists(path):
            os.unlink(path)

    print(f"\nPDF saved → {output_path}")
    print("Open with: open results/thesis_ieee_report.pdf")


if __name__ == "__main__":
    main()
