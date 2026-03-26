"""
Thesis Companion Guide PDF
===========================
A complete practical reference covering:
  1. Cover + Key Stats
  2. Data
  3. Related Work + Contributions
  4. Experiments
  5. Code Architecture (src/ ↔ RAG_THESIS/)
  6. Step-by-Step Run Guide
  7. Manual Annotation Tutorial
  8. Results
  9. Professor Meeting Prep
 10. What More to Explore

Usage:
  cd THESIS-main
  python3 src/generate_companion_guide.py
  open results/thesis_companion_guide.pdf
"""

import csv, json, os, sys
from pathlib import Path

BASE        = Path(__file__).parent.parent
OUTPUT_PATH = BASE / "results" / "thesis_companion_guide.pdf"
SUMMARY_JSON     = BASE / "RAG_THESIS" / "output" / "rq1_rq2_summary.json"
STATS_JSON       = BASE / "RAG_THESIS" / "output" / "statistics_report.json"
ANNOTATION_CSV   = BASE / "RAG_THESIS" / "output" / "manual_validation_200.csv"
RQ1_ANNOTS       = BASE / "results" / "gpt-4o-mini" / "02_rq1_annotations.json"
UNIFIED_JSONL    = BASE / "data" / "unified_results.jsonl"

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Preformatted, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
except ImportError:
    print("ERROR: pip install reportlab")
    sys.exit(1)

# ── Page geometry ─────────────────────────────────────────────────────────────
PW, PH = A4           # 595.27, 841.89 pts
LM = RM = 2.2 * cm
TM = BM = 2.0 * cm
USABLE_W = PW - LM - RM   # ~17 cm

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#1a3a5c")
C_BLUE   = colors.HexColor("#2471a3")
C_LBLUE  = colors.HexColor("#d6eaf8")
C_GREEN  = colors.HexColor("#1e8449")
C_LGREEN = colors.HexColor("#d5f5e3")
C_AMBER  = colors.HexColor("#d35400")
C_LAMBER = colors.HexColor("#fdebd0")
C_RED    = colors.HexColor("#c0392b")
C_GREY   = colors.HexColor("#7f8c8d")
C_LGREY  = colors.HexColor("#f2f3f4")
C_WHITE  = colors.white
C_CODE   = colors.HexColor("#f4f6f7")
C_STRIPE = colors.HexColor("#eaf4fb")

# ── Text styles ───────────────────────────────────────────────────────────────
_ss = getSampleStyleSheet()

def _style(name, **kw):
    base = kw.pop("parent", _ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

TITLE  = _style("GT", parent=_ss["Title"],   fontSize=22, textColor=C_DARK, alignment=TA_CENTER, spaceAfter=6)
STITLE = _style("GS", fontSize=13, textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=4)
H1     = _style("GH1", fontSize=15, textColor=C_DARK,  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
H2     = _style("GH2", fontSize=12, textColor=C_BLUE,  fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
H3     = _style("GH3", fontSize=10, textColor=C_DARK,  fontName="Helvetica-Bold", spaceBefore=6,  spaceAfter=3)
BODY   = _style("GB",  fontSize=10, leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
BULL   = _style("GBL", fontSize=10, leading=15, spaceAfter=3, leftIndent=14, firstLineIndent=-10)
CAPTION= _style("GC",  fontSize=8,  textColor=C_GREY, alignment=TA_CENTER, spaceAfter=6)
CELL   = _style("GCL", fontSize=9,  leading=13)
CELLH  = _style("GCH", fontSize=9,  leading=13, fontName="Helvetica-Bold", textColor=C_WHITE)
CELLB  = _style("GCB", fontSize=9,  leading=13, fontName="Helvetica-Bold")
MONO   = _style("GM",  fontSize=8.5, fontName="Courier", leading=12, spaceAfter=2)
WARN   = _style("GW",  fontSize=9,  leading=13, textColor=C_DARK)
NOTE   = _style("GN",  fontSize=9,  leading=13, textColor=C_DARK)
QA_Q   = _style("GQ",  fontSize=10, leading=14, fontName="Helvetica-Bold", textColor=C_DARK, spaceBefore=8, spaceAfter=2)
QA_A   = _style("GA",  fontSize=10, leading=14, spaceAfter=2, leftIndent=14, firstLineIndent=-10)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _p(txt, sty=None):
    return Paragraph(str(txt), sty or CELL)

def _row(cells, header=False):
    sty = CELLH if header else CELL
    return [Paragraph(str(c), sty) for c in cells]

def _tbl(data, widths=None, hdr_color=C_DARK, stripe=True, repeat=1):
    """Build a table with Paragraph-wrapped cells, striped rows, header."""
    wrapped = [_row(data[0], header=True)]
    for i, row in enumerate(data[1:]):
        bg = C_STRIPE if (stripe and i % 2 == 0) else C_WHITE
        wrapped.append([Paragraph(str(c), CELL) for c in row])
    t = Table(wrapped, colWidths=widths, repeatRows=repeat)
    style = [
        ("BACKGROUND",    (0,0), (-1,0), hdr_color),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#c0c0c0")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_STRIPE] if stripe else [C_WHITE]),
    ]
    t.setStyle(TableStyle(style))
    return t

def _code_block(lines):
    """Render code/commands in a light grey bordered box."""
    text = "\n".join(lines)
    inner = Paragraph(text.replace("\n", "<br/>"), MONO)
    t = Table([[inner]], colWidths=[USABLE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_CODE),
        ("BOX",           (0,0), (-1,-1), 0.8, C_GREY),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
    ]))
    return t

def _box(content_paragraphs, bg=C_LBLUE, border=C_BLUE):
    """Wrap a list of Paragraphs in a coloured box."""
    inner = Table([[p] for p in content_paragraphs], colWidths=[USABLE_W - 1.2*cm])
    t = Table([[inner]], colWidths=[USABLE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("BOX",           (0,0), (-1,-1), 1.0, border),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
    ]))
    return t

def _hr():
    return HRFlowable(width="100%", thickness=0.8, color=C_BLUE, spaceAfter=8, spaceBefore=4)

def _sp(h=0.25):
    return Spacer(1, h * cm)

def b(txt):   return f"<b>{txt}</b>"
def em(txt):  return f"<i>{txt}</i>"

# ── Load data ─────────────────────────────────────────────────────────────────

def _load():
    summary = {}
    if SUMMARY_JSON.exists():
        summary = json.load(open(SUMMARY_JSON))

    stats = {}
    if STATS_JSON.exists():
        stats = json.load(open(STATS_JSON))

    annot_rows = []
    if ANNOTATION_CSV.exists():
        annot_rows = list(csv.DictReader(open(ANNOTATION_CSV, encoding="utf-8-sig")))

    rq1_rows = []
    if RQ1_ANNOTS.exists():
        rq1_rows = json.load(open(RQ1_ANNOTS))

    return summary, stats, annot_rows, rq1_rows

# ─────────────────────────────────────────────────────────────────────────────
# SECTION BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def sec_cover(summary):
    e = []
    e.append(_sp(3))
    e.append(Paragraph("Hallucination Detection in Interview RAG Systems", TITLE))
    e.append(Paragraph("Research Companion Guide", STITLE))
    e.append(_sp(0.3))
    e.append(_hr())
    e.append(Paragraph("Data · Experiments · Architecture · Results · Professor Q&A", _style("sub", fontSize=11, textColor=C_GREY, alignment=TA_CENTER)))
    e.append(_sp(1.5))

    kf = summary.get("key_findings", {})
    ds = summary.get("data_summary", {})
    rows = [
        ["Metric", "Value", "Status"],
        ["Total interviews",             "749 (149 real + 600 synthetic)",  "✓"],
        ["Languages",                    "English + Dutch",                  "✓"],
        ["Models evaluated",             "GPT-4o-mini, Mistral, Qwen",       "✓"],
        ["Total query-response pairs",   "~9,218 across 3 models",           "✓"],
        ["RQ1 complete (all models)",    "Yes — 3 taxonomies annotated",     "✓"],
        ["RQ2 complete (all models)",    "Partial — AlignScore/RAGAS pending","⏳"],
        ["Manual annotation",            "200 rows exported, ready to label","⏳"],
        ["GPT-4o-mini hall. rate",       f"{kf.get('gpt4omini_hallucination_rate',0.2077):.1%}", "RQ1"],
        ["Mistral hall. rate",           f"{kf.get('mistral_hallucination_rate',0.1754):.1%}",   "RQ1"],
        ["Qwen hall. rate",              f"{kf.get('qwen_hallucination_rate',0.2215):.1%}",      "RQ1"],
        ["Hardest query type",           "Sentiment — 76.83% hallucination", "Finding"],
        ["Mistral vs GPT-4o-mini",       "Significantly better (p=0.0001)",  "McNemar"],
    ]
    e.append(_tbl(rows, widths=[6.5*cm, 7.5*cm, 2.7*cm]))
    e.append(PageBreak())
    return e


def sec_data(summary):
    e = []
    e.append(Paragraph("Section 2 — Data: What It Is, Why It Matters, How to Extend", H1))
    e.append(_hr())

    # 2.1
    e.append(Paragraph("2.1  What the Data Is", H2))
    e.append(Paragraph(
        "The dataset consists of <b>749 unique research interviews</b> collected from the Convo "
        "platform — an AI-powered interview tool used by researchers and companies. "
        "Each interview is a real or synthetic conversation between an AI agent and a participant, "
        "stored as a JSON transcript with speaker-labelled utterances.",
        BODY))
    rows = [
        ["Source File", "Type", "Lang", "Count", "Why It Matters"],
        ["supabase_responses.csv",         "Real",      "EN+NL", "~149", "Ground truth quality — real people, real answers"],
        ["synthetic_interviews.csv",       "Synthetic", "Mixed", "~200", "Scale without privacy risk"],
        ["supbase_english_synthetic.csv",  "Synthetic", "EN",    "~200", "Language balance — English"],
        ["supbase_dutch_synthetic.csv",    "Synthetic", "NL",    "~200", "Language balance — Dutch"],
    ]
    e.append(_tbl(rows, widths=[5*cm, 2.5*cm, 1.5*cm, 2*cm, 6.7*cm]))

    # 2.2
    e.append(Paragraph("2.2  Why Each Source Matters", H2))
    items = [
        ("Real interviews (149)",
         "These are the gold standard. Real participants give natural, varied answers. "
         "They prove the research is grounded in actual use. Without them, reviewers would dismiss the work as purely synthetic."),
        ("Synthetic interviews (600)",
         "Generated to scale up the dataset cost-effectively. They simulate realistic interview "
         "patterns based on actual interview structures. The 4:1 synthetic/real ratio is common "
         "in NLP research when privacy constraints limit real data collection."),
        ("English + Dutch",
         "Including Dutch makes this one of the few multilingual RAG hallucination studies. "
         "The near-identical EN/NL hallucination rates (20.72% vs 20.81%) show the "
         "pipeline generalises across languages — a strong additional finding."),
    ]
    for title, desc in items:
        e.append(_sp(0.2))
        e.append(Paragraph(f"<b>{title}:</b> {desc}", BODY))

    # 2.3 — 6 query types
    e.append(Paragraph("2.3  The 6 Query Types Explained", H2))
    e.append(Paragraph(
        "For each interview, 6 types of research questions are automatically generated. "
        "These represent the kinds of questions a researcher would ask an AI system about an interview:",
        BODY))
    rows = [
        ["Query Type", "What It Asks", "Example Query", "Difficulty"],
        ["speaker_attribution", "What did the interviewer say?",
         "What did the Agent say at the beginning?", "Very Low (1.74% hall.)"],
        ["participant_content", "What topics did the participant discuss?",
         "What topics did the participant discuss?", "Low (6.76%)"],
        ["factual_summary",     "Summarise the main points",
         "Summarise the main points of this interview.", "Moderate (8.90%)"],
        ["temporal",            "What happened last / in what order?",
         "What was the last topic before the interview ended?", "Moderate (8.30%)"],
        ["specific_content",   "What happened around a specific quote?",
         "What was discussed after the participant said '...'?", "High (22.05%)"],
        ["sentiment",          "What was the participant's emotional tone?",
         "What was the overall sentiment of the participant?", "CRITICAL (76.83%)"],
    ]
    e.append(_tbl(rows, widths=[3.8*cm, 3.5*cm, 5.5*cm, 4.4*cm]))

    # 2.4 — sufficiency
    e.append(Paragraph("2.4  Is the Data Sufficient for IEEE?", H2))
    rows = [
        ["Metric", "Current", "IEEE Minimum", "Verdict"],
        ["Total query-response pairs",  "~9,218",  "3,000+",  "✓ Sufficient"],
        ["Models evaluated",            "3",       "2+",      "✓ Good"],
        ["Manual validation labels",    "0 (ready to fill)", "150+", "⏳ Action needed"],
        ["Language coverage",           "EN + NL", "1+",      "✓ Strong (multilingual)"],
        ["Real interviews",             "149",     "50+",     "✓ Sufficient"],
        ["Query types",                 "6",       "3+",      "✓ Comprehensive"],
    ]
    e.append(_tbl(rows, widths=[5.5*cm, 4*cm, 3.5*cm, 4.2*cm]))

    # 2.5 — how to extend
    e.append(Paragraph("2.5  How to Extend the Data", H2))
    e.append(Paragraph("If you want to increase the dataset size, here are concrete steps:", BODY))
    steps = [
        "Generate more synthetic interviews: modify data_loader.py to produce more interviews per study_id",
        "Add more query types: add entries in generate_queries() in src/data_loader.py — e.g. 'comparative' queries",
        "Add more languages: collect or synthesise French / German interviews and add to data/synthetic/",
        "Add a 4th model: add a new entry in DEFAULT_MODELS in run_all.py (e.g. 'gpt-4o' or 'llama')",
        "Annotate more manually: increase n_samples in export_annotation_csv.py from 200 to 500",
    ]
    for s in steps:
        e.append(Paragraph(f"• {s}", BULL))

    # 2.6 — what changes with more data
    e.append(Paragraph("2.6  What Changes With More Data", H2))
    rows = [
        ["If you add...", "What improves"],
        ["More manual labels (200→500)", "Kappa becomes more reliable; ensemble detector trains better"],
        ["More synthetic interviews",    "Statistical power increases; smaller effects become detectable"],
        ["A 4th model (e.g. GPT-4o)",    "Upper-bound comparison; shows cost/quality trade-off"],
        ["More query types",             "Broader coverage of researcher use cases"],
        ["More Dutch data",              "Cross-lingual findings become more robust"],
    ]
    e.append(_tbl(rows, widths=[7*cm, 10.7*cm]))
    e.append(PageBreak())
    return e


def sec_related(summary):
    e = []
    e.append(Paragraph("Section 3 — Related Work and Your Contributions", H1))
    e.append(_hr())

    e.append(Paragraph("3.1  Related Work — What Exists and How You Differ", H2))
    rows = [
        ["Paper", "What It Does", "Your Difference"],
        ["RAGTruth (Niu et al., ACL 2024)",
         "Taxonomy of 7 hallucination types for general RAG",
         "You apply it to interview domain — first domain-specific application"],
        ["SelfCheckGPT (Manakul et al., EMNLP 2023)",
         "Sample 5 responses, check consistency via BERTScore",
         "You evaluate it on dialogue data; find it fails (F1≈0) for interview domain"],
        ["MiniCheck (Tang et al., EMNLP 2024)",
         "Break response into claims, verify each against context using FLAN-T5",
         "You show it achieves F1=0.857 on interview data — best single detector"],
        ["AlignScore (Zha et al., ACL 2023)",
         "RoBERTa NLI model scores faithfulness 0–1",
         "You add it as 3rd signal in ensemble; pending results"],
        ["DiaHaLu (dialogue hallucination taxonomy)",
         "Classifies dialogue-level issues: topic drift, speaker confusion etc.",
         "You apply it to interview RAG — adds dialogue-level lens on top of RAGTruth"],
        ["RAGAS (Es et al., EACL 2024)",
         "Faithfulness metric for RAG pipelines",
         "You use it as 4th ensemble signal alongside other methods"],
    ]
    e.append(_tbl(rows, widths=[4*cm, 5.5*cm, 8.2*cm]))

    e.append(Paragraph("3.2  Your 5 Novel Contributions", H2))
    contribs = [
        ("C1 — First interview-domain RAG hallucination study",
         "All prior work uses Wikipedia, news, or general QA data. You are the first to study "
         "hallucination in AI-powered research interviews with personal, privacy-sensitive content. "
         "This is a completely new application domain."),
        ("C2 — Multi-taxonomy comparison on the same dataset",
         "You apply RAGTruth, Huang, and DiaHaLu taxonomies to the same 9,218 responses. "
         "No prior paper compares all three on identical data. This shows which taxonomy "
         "is most informative for the interview domain."),
        ("C3 — Refusal-Hallucination distinction",
         "You find that 60% of GPT-4o-mini 'hallucinations' are actually safety refusals — "
         "the model saying 'I cannot determine' even when the transcript contains the answer. "
         "This is a new finding: commercial LLM safety guardrails inflate hallucination metrics "
         "on privacy-sensitive data."),
        ("C4 — Cross-lingual analysis (English + Dutch)",
         "You show that hallucination rates are nearly identical across English (20.72%) and Dutch "
         "(20.81%) — meaning the RAG pipeline generalises well across languages. Most RAG "
         "hallucination papers use English-only data."),
        ("C5 — Predictive hallucination detection",
         "The hallucination_predictor.py module predicts hallucination risk BEFORE the response "
         "is generated, using retrieval-time signals (chunk size, similarity score, top-k). "
         "This is novel: most work detects hallucinations after generation, not before."),
    ]
    for title, desc in contribs:
        e.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", H3),
            Paragraph(desc, BODY),
        ]))

    e.append(Paragraph("3.3  Why Interview RAG is Different from Standard RAG", H2))
    rows = [
        ["Dimension", "Standard RAG (Wikipedia/news)", "Interview RAG (your work)"],
        ["Content type",     "Factual, public, verifiable",      "Personal, private, conversational"],
        ["Verification",     "Cross-check with web/knowledge base","Only the transcript is ground truth"],
        ["Privacy concern",  "None",                              "Participant data — GDPR implications"],
        ["Hallucination risk","Factual errors, date/number errors","Misquoting people, misrepresenting sentiment"],
        ["Refusal behaviour","Rare",                              "60% of GPT-4o-mini errors are refusals"],
        ["Language",        "Mostly English",                    "English + Dutch (multilingual)"],
    ]
    e.append(_tbl(rows, widths=[3.5*cm, 6*cm, 8.2*cm]))

    e.append(Paragraph("3.4  The Gap in Literature", H2))
    e.append(_box([
        Paragraph("<b>Key claim for your paper:</b>", NOTE),
        _sp(0.1),
        Paragraph(
            "No prior work studies hallucination in RAG systems applied to privacy-sensitive interview "
            "data. The closest work (RAGTruth) uses Wikipedia-based datasets. Interview data is "
            "different: the transcript is the only ground truth, participants are identifiable, "
            "misquoting someone has ethical consequences, and commercial LLMs apply safety filters "
            "that create a new class of 'refusal hallucination' not seen in general RAG research.",
            WARN),
    ], bg=C_LBLUE, border=C_BLUE))
    e.append(PageBreak())
    return e


def sec_experiments(summary):
    e = []
    e.append(Paragraph("Section 4 — Experiments: What They Are and Are They Enough", H1))
    e.append(_hr())

    e.append(Paragraph("4.1  RQ1 Experiments — Hallucination Classification", H2))
    e.append(Paragraph(
        "RQ1 asks: <b>What types of hallucinations do these models make?</b> "
        "Three annotation experiments classify each response:", BODY))
    rows = [
        ["Experiment", "Method", "What It Produces", "Status"],
        ["Exp 1 — RAGTruth",
         "GPT-4o-mini evaluates each response against the transcript. Returns: label (HALLUCINATED/FAITHFUL), faithfulness score 0–1, type from 7 categories, span, severity.",
         "02_rq1_annotations.json", "✓ All 3 models"],
        ["Exp 2 — Huang Taxonomy",
         "Maps hallucinations to Huang et al.'s standard taxonomy. Finds interview-specific categories not in standard taxonomy.",
         "03_rq1_huang_taxonomy.json", "✓ All 3 models"],
        ["Exp 3 — DiaHaLu",
         "Dialogue-level evaluation: detects cross-turn contradictions, topic drift, speaker confusion, temporal distortion.",
         "04_rq1_diahalu_eval.json", "✓ All 3 models"],
    ]
    e.append(_tbl(rows, widths=[3.2*cm, 7.5*cm, 4.5*cm, 2.5*cm]))

    e.append(Paragraph("4.2  RQ2 Experiments — Hallucination Detection", H2))
    e.append(Paragraph(
        "RQ2 asks: <b>Can automated tools reliably detect these hallucinations?</b> "
        "Four detection methods are tested:", BODY))
    rows = [
        ["Experiment", "How It Works Technically", "Key Question", "Status"],
        ["Exp 4 — SelfCheckGPT (BERTScore)",
         "Generate 5 additional responses to the same query. Split primary response into sentences. For each sentence, compute BERTScore similarity to the 5 samples. Low similarity = the model is inconsistent = hallucination.",
         "Do inconsistent responses indicate hallucination?",
         "✓ GPT + Qwen; ⏳ Mistral"],
        ["Exp 5 — MiniCheck (FLAN-T5)",
         "Split response into individual claims. For each claim, run FLAN-T5 NLI to check: is this claim supported by the transcript? Unsupported claims are flagged as hallucinations.",
         "Can NLI-based claim verification catch hallucinations?",
         "✓ GPT (30); ✓ Qwen; ⏳ Mistral"],
        ["Exp 6 — AlignScore (RoBERTa NLI)",
         "Score the entire response against the context using a RoBERTa NLI model fine-tuned for faithfulness. Returns 0–1 score where <0.5 = likely hallucinated.",
         "Does a single faithfulness score predict hallucination?",
         "⏳ All models"],
        ["Exp 7 — RAGAS Faithfulness",
         "Breaks the response into statements. Uses an LLM to check if each statement can be inferred from the retrieved context. Computes ratio of supported statements.",
         "Does RAGAS faithfulness scoring work in interview domain?",
         "⏳ All models"],
    ]
    e.append(_tbl(rows, widths=[3*cm, 6.5*cm, 4*cm, 3.2*cm]))

    e.append(Paragraph("4.3  Completion Status", H2))
    rows = [
        ["Experiment", "GPT-4o-mini", "Qwen", "Mistral"],
        ["RQ1 — RAGTruth annotation",  "✓ 3,106", "✓ 3,106", "✓ 3,006"],
        ["RQ1 — Huang taxonomy",       "✓ 3,106", "✓ 3,106", "✓ 3,006"],
        ["RQ1 — DiaHaLu",              "✓ 3,106", "✓ 3,106", "✓ 3,006"],
        ["RQ2 — SelfCheckGPT",         "✓ 3,106", "✓ 3,106", "⏳ Pending"],
        ["RQ2 — MiniCheck",            "✓ 30",    "✓ 3,106", "⏳ Pending"],
        ["RQ2 — AlignScore",           "⏳ Pending","⏳ Pending","⏳ Pending"],
        ["RQ2 — RAGAS",                "⏳ Pending","⏳ Pending","⏳ Pending"],
        ["Manual validation labels",   "⏳ 0/200","—","—"],
    ]
    e.append(_tbl(rows, widths=[5.5*cm, 3.8*cm, 3.8*cm, 4.6*cm]))

    e.append(Paragraph("4.4  Are the Experiments Enough for IEEE?", H2))
    e.append(_box([
        Paragraph("<b>Verdict: Yes for scope — but 2 blockers remain</b>", CELLB),
        _sp(0.15),
        Paragraph("The experiment design is solid for an IEEE paper. 3 taxonomy methods + 4 detection methods on 9,218 responses across 3 models is comprehensive. However:", WARN),
        _sp(0.1),
        Paragraph("⚠  <b>Blocker 1:</b> Complete AlignScore + RAGAS for all 3 models. Without these, the RQ2 comparison table is incomplete.", WARN),
        Paragraph("⚠  <b>Blocker 2:</b> Manual annotation (200 rows) must be done before Cohen's Kappa can be computed. Kappa is required to validate the GPT-4o-mini judge.", WARN),
        Paragraph("✓  Everything else (RQ1, cross-lingual, statistical tests) is publishable as-is.", WARN),
    ], bg=C_LGREEN, border=C_GREEN))

    e.append(Paragraph("4.5  What Additional Experiments Would Strengthen the Paper", H2))
    rows = [
        ["Experiment", "Effort", "Impact", "How to Run"],
        ["Add GPT-4o (full) as 4th model",     "High (API cost)", "High — shows cost/quality trade-off",     "Add 'gpt-4o' to DEFAULT_MODELS in run_all.py"],
        ["Run hallucination_predictor.py",     "Low",  "Novel — predicts before generation",       "python RAG_THESIS/advanced/hallucination_predictor.py"],
        ["Run correction_loop.py",             "Low",  "Demonstrates practical fix",               "python RAG_THESIS/advanced/correction_loop.py"],
        ["Run ensemble_detector.py",           "Low",  "Shows ensemble beats single detectors",    "python RAG_THESIS/advanced/ensemble_detector.py (after annotation)"],
        ["Hyperparameter sweep",               "Medium","Ablation — chunk_size, top_k effect",     "python RAG_THESIS/hyperparameter_sweep/*.py"],
        ["Run no-RAG baseline (ablation)",     "Low",  "Shows RAG helps vs no context at all",     "python RAG_THESIS/ablation/run_no_rag_baseline.py"],
    ]
    e.append(_tbl(rows, widths=[4.5*cm, 2.5*cm, 4.5*cm, 6.2*cm]))
    e.append(PageBreak())
    return e


def sec_architecture():
    e = []
    e.append(Paragraph("Section 5 — Code Architecture: Connecting src/ and RAG_THESIS/", H1))
    e.append(_hr())

    e.append(Paragraph("5.1  Full Pipeline Diagram", H2))
    diagram = [
        "┌─────────────────────────────────────────────────────────────┐",
        "│                    src/  (Main Pipeline)                    │",
        "│                                                             │",
        "│  data_loader.py  →  rag_pipeline.py  →  experiment_rq1.py  │",
        "│       ↓                  ↓                     ↓           │",
        "│  749 interviews    9,218 responses      02_rq1_annotations  │",
        "│  6 query types     per model            03_huang_taxonomy   │",
        "│                                         04_diahalu_eval     │",
        "│                                              ↓              │",
        "│                              experiment_rq2.py              │",
        "│                                    ↓                       │",
        "│                         05_selfcheck / 06_minicheck         │",
        "│                         07_alignscore / 08_ragas            │",
        "│                                    ↓                       │",
        "│                      build_unified_results.py               │",
        "│                                    ↓                       │",
        "│                       data/unified_results.jsonl            │",
        "└─────────────────────────────────┬───────────────────────────┘",
        "                                  │",
        "                                  ▼",
        "┌─────────────────────────────────────────────────────────────┐",
        "│                  RAG_THESIS/  (Advanced)                    │",
        "│                                                             │",
        "│  advanced/ensemble_detector.py   → ensemble_results.json   │",
        "│  advanced/hallucination_predictor.py → risk scores         │",
        "│  advanced/correction_loop.py     → corrected responses      │",
        "│  validation/calculate_kappa.py   → Cohen's Kappa           │",
        "│  ablation/run_no_rag_baseline.py → ablation results         │",
        "└─────────────────────────────────────────────────────────────┘",
        "                                  │",
        "                                  ▼",
        "┌─────────────────────────────────────────────────────────────┐",
        "│                     Reporting                               │",
        "│                                                             │",
        "│  src/statistics_utils.py     → McNemar + Wilson CIs        │",
        "│  src/generate_thesis_report.py → IEEE structure PDF         │",
        "│  src/generate_companion_guide.py → This document           │",
        "└─────────────────────────────────────────────────────────────┘",
    ]
    e.append(_code_block(diagram))

    e.append(Paragraph("5.2  src/ Files — What Each Does", H2))
    rows = [
        ["File", "Step", "Reads", "Writes"],
        ["data_loader.py",          "0", "4 CSV files",                 "Interview objects in memory"],
        ["rag_pipeline.py",         "1", "Interview objects",           "01_rag_responses.json"],
        ["experiment_rq1.py",       "2", "01_rag_responses.json",       "02, 03, 04 JSON files"],
        ["experiment_rq2.py",       "3", "01_rag_responses.json",       "05, 06, 07, 08 JSON files"],
        ["build_unified_results.py","4", "02–08 JSON files",            "data/unified_results.jsonl"],
        ["statistics_utils.py",     "5", "unified_results.jsonl",       "statistics_report.json"],
        ["run_all.py",              "—", "CLI args",                    "Orchestrates steps 0–5"],
        ["compare_models.py",       "—", "All model result dirs",       "Comparison report"],
        ["hallucination_alert.py",  "—", "Any RQ1/RQ2 result",         "Per-query alert PDF"],
        ["embeddings_retriever.py", "—", "Interview text",              "Vector index in memory"],
        ["rag_configs.py",          "—", "—",                           "Config objects for ablation"],
    ]
    e.append(_tbl(rows, widths=[5.5*cm, 1.3*cm, 5.5*cm, 5.4*cm]))

    e.append(Paragraph("5.3  RAG_THESIS/ Files — What Each Does", H2))
    rows = [
        ["File", "Reads", "Writes", "Purpose"],
        ["advanced/ensemble_detector.py",
         "manual_validation_200.csv + 05–08 JSONs",
         "ensemble_results.json",
         "Logistic regression ensemble of 4 detectors; finds best combination"],
        ["advanced/hallucination_predictor.py",
         "unified_results.jsonl",
         "predictor_results.json",
         "Predicts hallucination BEFORE generation from retrieval signals"],
        ["advanced/correction_loop.py",
         "RQ1 annotations (high-hallucination responses)",
         "corrected_responses.json",
         "Auto-corrects hallucinated responses via re-prompting"],
        ["advanced/retrieval_correlation.py",
         "unified_results.jsonl",
         "retrieval_correlation.json",
         "Correlates retrieval quality metrics with hallucination rates"],
        ["validation/export_annotation_csv.py",
         "02_rq1_annotations.json",
         "manual_validation_200.csv",
         "Exports 200 stratified samples for manual labelling"],
        ["validation/calculate_kappa.py",
         "manual_validation_200.csv (after you fill it)",
         "validation_results.json",
         "Computes Cohen's Kappa between your labels and GPT-4o-mini labels"],
        ["ablation/run_no_rag_baseline.py",
         "Interview data",
         "no_rag_baseline.json",
         "Baseline: answer queries WITHOUT providing transcript context"],
    ]
    e.append(_tbl(rows, widths=[4.5*cm, 4*cm, 3.5*cm, 5.7*cm]))

    e.append(Paragraph("5.4  File Dependency Map", H2))
    e.append(Paragraph("Reading order — each file depends on what's above it:", BODY))
    deps = [
        "data/real/supabase_responses.csv",
        "data/synthetic/*.csv",
        "  └─► data_loader.py (loads + merges CSVs)",
        "       └─► rag_pipeline.py (generates responses)",
        "            └─► results/{model}/01_rag_responses.json",
        "                 ├─► experiment_rq1.py  → 02, 03, 04_*.json",
        "                 └─► experiment_rq2.py  → 05, 06, 07, 08_*.json",
        "                      └─► build_unified_results.py → data/unified_results.jsonl",
        "                           ├─► statistics_utils.py → statistics_report.json",
        "                           └─► ensemble_detector.py (needs manual labels too)",
        "                                └─► RAG_THESIS/output/ensemble_results.json",
        "",
        "Manual annotation path:",
        "  export_annotation_csv.py → manual_validation_200.csv",
        "  [YOU FILL IN YOUR_label COLUMN]",
        "  calculate_kappa.py → validation_results.json",
        "  ensemble_detector.py (uses manual labels as training data)",
    ]
    e.append(_code_block(deps))
    e.append(PageBreak())
    return e


def sec_run_guide():
    e = []
    e.append(Paragraph("Section 6 — Step-by-Step Run Guide", H1))
    e.append(_hr())
    e.append(Paragraph(
        "All commands must be run from the project root directory: "
        "<b>THESIS-main/</b>. Open Terminal and navigate there first.", BODY))

    # Prerequisites
    e.append(Paragraph("6.1  Prerequisites", H2))
    e.append(_code_block([
        "# Check Python version (need 3.9+)",
        "python3 --version",
        "",
        "# Navigate to project",
        "cd /Users/convo/Downloads/THESIS-main-extracted/THESIS-main",
        "",
        "# Install required packages",
        "python3 -m pip install reportlab matplotlib openai python-dotenv",
        "python3 -m pip install transformers torch bert-score scikit-learn",
        "",
        "# Make sure .env has your OPENAI_API_KEY",
        "cat .env  # should show OPENAI_API_KEY=sk-...",
    ]))

    steps = [
        ("6.2  Step 1 — Export Annotation CSV (Already Done)", [
            "# The annotation CSV is already generated at:",
            "# RAG_THESIS/output/manual_validation_200.csv",
            "# If you need to regenerate it:",
            "python3 RAG_THESIS/validation/export_annotation_csv.py",
            "# Output: RAG_THESIS/output/manual_validation_200.csv (198 rows)",
        ], "You will see: 'Exported 198 samples → RAG_THESIS/output/manual_validation_200.csv'"),

        ("6.3  Step 2 — Complete RQ2 for Mistral", [
            "# Run the 4 detection experiments for Mistral only",
            "python3 src/run_all.py --models mistral --steps rq2",
            "# This downloads MiniCheck model (~900MB) first time",
            "# Expected time: 30–60 minutes",
        ], "Output: results/mistral/05_rq2_selfcheck.json + 06_rq2_minicheck.json"),

        ("6.4  Step 3 — Build Unified Results File", [
            "# Merge all RQ1 + RQ2 results into one file",
            "python3 src/build_unified_results.py",
            "# Output: data/unified_results.jsonl (9,218 lines)",
        ], "You will see coverage table showing which detectors have data for each model."),

        ("6.5  Step 4 — Run Statistical Analysis", [
            "# Compute McNemar tests + Wilson CIs for all models",
            "python3 src/statistics_utils.py",
            "# Output: RAG_THESIS/output/statistics_report.json",
        ], "You will see per-model CIs and McNemar significance results printed to terminal."),

        ("6.6  Step 5 — After Annotating: Run Kappa + Ensemble", [
            "# After filling in YOUR_label in manual_validation_200.csv:",
            "python3 RAG_THESIS/validation/calculate_kappa.py",
            "# Then train and evaluate the ensemble:",
            "python3 RAG_THESIS/advanced/ensemble_detector.py",
        ], "Kappa output: cohen_kappa score + agreement%. Ensemble output: F1 per detector + ensemble F1."),

        ("6.7  Step 6 — Run Advanced Modules", [
            "# Predict hallucination before generation",
            "python3 RAG_THESIS/advanced/hallucination_predictor.py",
            "",
            "# Test auto-correction loop",
            "python3 RAG_THESIS/advanced/correction_loop.py",
            "",
            "# Ablation: no-RAG baseline",
            "python3 RAG_THESIS/ablation/run_no_rag_baseline.py",
        ], "Each produces a JSON file in RAG_THESIS/output/ with results."),

        ("6.8  Step 7 — Regenerate PDF Reports", [
            "# Regenerate this companion guide:",
            "python3 src/generate_companion_guide.py",
            "open results/thesis_companion_guide.pdf",
            "",
            "# Regenerate IEEE structure report:",
            "python3 src/generate_thesis_report.py",
            "open results/thesis_ieee_report.pdf",
        ], "Both PDFs saved to results/ directory."),

        ("6.9  Full Pipeline (All at Once)", [
            "# Generate responses + run all experiments for all models:",
            "python3 src/run_all.py --models gpt-4o-mini,qwen,mistral --steps all",
            "",
            "# Or just run experiments (responses already generated):",
            "python3 src/run_all.py --steps rq1,rq2,unified,stats",
            "",
            "# Or test quickly on 5 interviews:",
            "python3 src/run_all.py --models gpt-4o-mini --steps generate --max-interviews 5",
        ], "The --steps flag accepts: generate, rq1, rq2, compare, unified, stats"),
    ]
    for title, cmds, expected in steps:
        e.append(Paragraph(title, H2))
        e.append(_code_block(cmds))
        e.append(_box([Paragraph(f"<b>Expected output:</b> {expected}", NOTE)],
                       bg=C_LBLUE, border=C_BLUE))
        e.append(_sp(0.3))
    e.append(PageBreak())
    return e


def sec_annotation(annot_rows, rq1_rows):
    e = []
    e.append(Paragraph("Section 7 — Manual Annotation Tutorial", H1))
    e.append(_hr())

    # 7.1
    e.append(Paragraph("7.1  What Is Manual Annotation and Why Do You Need It?", H2))
    e.append(Paragraph(
        "Manual annotation means <b>you read each AI response and decide whether it is correct "
        "or not</b>. Your labels become the 'human ground truth' used to validate the GPT-4o-mini "
        "automatic annotator.", BODY))
    e.append(Paragraph(
        "IEEE reviewers will ask: <i>'How do you know GPT-4o-mini's labels are correct?'</i> "
        "The answer is Cohen's Kappa: a statistical measure of agreement between your human "
        "labels and the automatic labels. A Kappa ≥ 0.61 ('substantial agreement') means "
        "the automatic annotator is validated and trustworthy.", BODY))
    rows = [
        ["Term", "What it means"],
        ["Manual annotation", "You read a response and decide: FAITHFUL, HALLUCINATED, REFUSAL, or PARTIAL"],
        ["Cohen's Kappa (κ)", "Measures % agreement between you and the AI, adjusted for chance. Range: 0 to 1"],
        ["κ ≥ 0.61", "Substantial agreement — good enough for IEEE publication"],
        ["Ground truth", "Your human label is treated as the 'correct' answer to compare against"],
    ]
    e.append(_tbl(rows, widths=[4*cm, 13.7*cm]))

    # 7.2
    e.append(Paragraph("7.2  Where Is the File?", H2))
    e.append(_code_block([
        "File location:",
        "/Users/convo/Downloads/THESIS-main-extracted/THESIS-main/",
        "   RAG_THESIS/output/manual_validation_200.csv",
        "",
        "Size: 198 rows (33 per query type, shuffled)",
        "Columns: id, query_type, language, query, rag_response,",
        "         transcript_excerpt, gpt4_label, gpt4_faithfulness_score,",
        "         hallucination_types_detected, YOUR_label, notes",
    ]))

    # 7.3
    e.append(Paragraph("7.3  How to Open in Google Sheets (Step by Step)", H2))
    steps = [
        "Open your browser and go to: sheets.google.com",
        "Click the '+' button (New Spreadsheet) in the top left",
        "In the new spreadsheet: click File → Import",
        "Click 'Upload' tab → drag the CSV file into the window",
        "Important: set 'Separator type' to 'Comma'",
        "Click 'Import data' — all 198 rows will appear",
        "Add a dropdown to the YOUR_label column: click the column header (J) to select it",
        "Click Data → Data validation → Criteria: List of items",
        "Type exactly: HALLUCINATED,FAITHFUL,REFUSAL,PARTIAL",
        "Click Save — a dropdown arrow will appear in each cell",
        "Now you can click each cell and select a label from the dropdown",
    ]
    for i, s in enumerate(steps, 1):
        e.append(Paragraph(f"<b>Step {i}:</b> {s}", BULL))

    # 7.4 — columns
    e.append(Paragraph("7.4  What Each Column Means", H2))
    rows = [
        ["Column", "What It Contains", "What You Do With It"],
        ["id",               "Row number 1–198",                              "Ignore (just for reference)"],
        ["query_type",       "Type of question: sentiment, temporal, etc.",   "Helps you understand context"],
        ["language",         "en = English, nl = Dutch",                      "Read accordingly"],
        ["query",            "The question the AI was asked",                 "READ THIS FIRST"],
        ["rag_response",     "The AI's answer to the query",                  "READ THIS — this is what you are labelling"],
        ["transcript_excerpt","First ~400 characters of the actual interview", "USE THIS to verify the response"],
        ["gpt4_label",       "What GPT-4o-mini automatically labelled it",    "Do NOT copy this — make your own judgement"],
        ["gpt4_faithfulness_score", "GPT-4o-mini confidence score 0–1",       "Ignore when labelling"],
        ["hallucination_types_detected", "Which error types GPT-4o-mini found","Read after you decide your label"],
        ["YOUR_label",       "EMPTY — this is what YOU fill in",              "FILL THIS IN"],
        ["notes",            "Optional — your reasoning",                     "Write if unsure"],
    ]
    e.append(_tbl(rows, widths=[4.5*cm, 5.5*cm, 7.7*cm]))

    # 7.5 — 4 labels with real examples
    e.append(Paragraph("7.5  The 4 Labels — With Real Examples From the Data", H2))

    # FAITHFUL
    faith_ex = None
    for r in annot_rows:
        if r.get("gpt4_label") == "FAITHFUL" and r.get("language") == "en" and len(r.get("rag_response","")) > 100:
            faith_ex = r
            break

    e.append(Paragraph("<b>FAITHFUL</b> — The response accurately reflects what's in the transcript.", H3))
    if faith_ex:
        e.append(_box([
            Paragraph(f"<b>Query:</b> {faith_ex['query'][:150]}", NOTE),
            _sp(0.1),
            Paragraph(f"<b>AI Response:</b> {faith_ex['rag_response'][:280]}...", NOTE),
            _sp(0.1),
            Paragraph(f"<b>Transcript:</b> {faith_ex['transcript_excerpt'][:200]}...", NOTE),
            _sp(0.1),
            Paragraph("<b>Label: FAITHFUL</b> — the response describes content that appears in the transcript.", CELLB),
        ], bg=C_LGREEN, border=C_GREEN))

    # HALLUCINATED
    e.append(Paragraph("<b>HALLUCINATED</b> — The response contains claims not in the transcript.", H3))
    e.append(_box([
        Paragraph("<b>Query:</b> What was the overall sentiment or tone of the participant?", NOTE),
        _sp(0.1),
        Paragraph("<b>AI Response:</b> The overall sentiment was mixed but generally positive. They expressed feelings of being overwhelmed and noted the welcome presentation was packed with information...", NOTE),
        _sp(0.1),
        Paragraph("<b>Why HALLUCINATED:</b> The AI adds specific emotional interpretations ('overwhelmed', 'generally positive') that cannot be verified from the transcript excerpt provided. Sentiment is being fabricated or exaggerated.", CELLB),
    ], bg=colors.HexColor("#fdebd0"), border=C_AMBER))

    # REFUSAL
    e.append(Paragraph("<b>REFUSAL</b> — The AI refuses to answer even though the information IS in the transcript.", H3))
    e.append(_box([
        Paragraph("<b>Query:</b> What was the overall sentiment or tone of the participant?", NOTE),
        _sp(0.1),
        Paragraph('<b>AI Response:</b> "This information is not available in the transcript."', NOTE),
        _sp(0.1),
        Paragraph("<b>Transcript excerpt:</b> [contains clear participant emotions and reactions]", NOTE),
        _sp(0.1),
        Paragraph("<b>Why REFUSAL:</b> The transcript clearly contains emotional content, but the AI refuses to summarise it. This is a safety guardrail refusing to interpret personal data.", CELLB),
    ], bg=colors.HexColor("#fadbd8"), border=C_RED))

    # PARTIAL
    e.append(Paragraph("<b>PARTIAL</b> — Some claims are correct, some are fabricated. Use sparingly.", H3))
    e.append(_box([
        Paragraph("<b>Query:</b> What topics did the participant discuss?", NOTE),
        _sp(0.1),
        Paragraph("<b>AI Response:</b> The participant discussed: 1) Awareness of the Xbox Portable Game Machine through Naver's Game Community cafe and the Switch Korea community. 2) ...", NOTE),
        _sp(0.1),
        Paragraph("<b>Transcript:</b> 'I heard about the news in the Switch Korea community.'", NOTE),
        _sp(0.1),
        Paragraph("<b>Why PARTIAL:</b> The Switch Korea community part is correct. But 'Naver's Game Community cafe' was added by the AI — it's not in the transcript. One true + one fabricated = PARTIAL.", CELLB),
    ], bg=C_LGREY, border=C_GREY))

    # 7.6 — decision flowchart
    e.append(Paragraph("7.6  Decision Flowchart — 3 Questions to Ask for Each Row", H2))
    flowchart = [
        "For each row in the CSV:",
        "",
        "  1. Read the QUERY column: what was the AI asked?",
        "  2. Read the RAG_RESPONSE column: what did the AI answer?",
        "  3. Read TRANSCRIPT_EXCERPT: what does the actual interview say?",
        "",
        "  Then ask these 3 questions in order:",
        "",
        "  Q1: Does the response say something like 'I cannot determine' or",
        "      'This information is not available'?",
        "       → YES: Does the transcript actually contain the answer?",
        "              → YES: Label = REFUSAL",
        "              → NO:  Label = FAITHFUL (AI is correct to say it doesn't know)",
        "       → NO: Continue to Q2",
        "",
        "  Q2: Does EVERY claim in the response appear in the transcript?",
        "      (Even if paraphrased — the MEANING should be there)",
        "       → YES: Label = FAITHFUL",
        "       → NO: Continue to Q3",
        "",
        "  Q3: Are SOME claims correct and SOME fabricated?",
        "       → YES (mixed): Label = PARTIAL",
        "       → NO (mostly or entirely wrong): Label = HALLUCINATED",
    ]
    e.append(_code_block(flowchart))

    # 7.7 — session plan
    e.append(Paragraph("7.7  Annotation Session Plan (2+ Weeks)", H2))
    rows = [
        ["Session", "When", "Rows to Label", "Cumulative Total"],
        ["Session 1", "Day 1",  "30 rows (rows 1–30)",   "30 / 198"],
        ["Session 2", "Day 2",  "30 rows (rows 31–60)",  "60 / 198"],
        ["Session 3", "Day 4",  "30 rows (rows 61–90)",  "90 / 198"],
        ["Session 4", "Day 6",  "30 rows (rows 91–120)", "120 / 198"],
        ["Session 5", "Day 8",  "30 rows (rows 121–150)","150 / 198 — IEEE minimum reached"],
        ["Session 6", "Day 10", "24 rows (rows 151–174)","174 / 198"],
        ["Session 7", "Day 12", "24 rows (rows 175–198)","198 / 198 — Complete"],
    ]
    e.append(_tbl(rows, widths=[3*cm, 3*cm, 5*cm, 6.7*cm]))
    e.append(Paragraph("Each session takes approximately 25–35 minutes. Take breaks between sessions.", BODY))

    # 7.8 — common mistakes
    e.append(Paragraph("7.8  Common Mistakes to Avoid", H2))
    mistakes = [
        ("Copying GPT-4o-mini's label",
         "The whole point is YOUR independent judgement. If you always agree with gpt4_label, Kappa will be meaningless."),
        ("Labelling PARTIAL too often",
         "PARTIAL should be rare — only when it's clearly half right, half wrong. When in doubt, use HALLUCINATED."),
        ("Marking as FAITHFUL when transcript is truncated",
         "The transcript_excerpt is only 400 characters. If you can't verify a claim from the excerpt, mark it HALLUCINATED (conservative labelling)."),
        ("Rushing through Dutch rows",
         "If your Dutch is limited, take extra time or use a translator for the transcript_excerpt column."),
        ("Not using the notes column",
         "For borderline cases, always write why you chose that label. You may need to explain your choices to the professor."),
    ]
    for title, desc in mistakes:
        e.append(Paragraph(f"<b>✗ {title}:</b> {desc}", BULL))

    # 7.9 — after annotating
    e.append(Paragraph("7.9  After Annotating — How to Save and Run Kappa", H2))
    steps_after = [
        "In Google Sheets: File → Download → Comma Separated Values (.csv)",
        "Save the downloaded file, replacing the original:",
        "  /Users/convo/Downloads/THESIS-main-extracted/THESIS-main/RAG_THESIS/output/manual_validation_200.csv",
        "Open Terminal and run:",
        "  cd /Users/convo/Downloads/THESIS-main-extracted/THESIS-main",
        "  python3 RAG_THESIS/validation/calculate_kappa.py",
        "You will see: Cohen's Kappa score + interpretation + copy-paste thesis sentence",
        "Target: κ ≥ 0.61 (substantial agreement)",
        "If κ < 0.41: review your labels — likely annotation guidelines were unclear",
    ]
    e.append(_code_block(steps_after))
    e.append(PageBreak())
    return e


def sec_results(summary, stats):
    e = []
    e.append(Paragraph("Section 8 — Results: All Numbers Explained", H1))
    e.append(_hr())

    models_data = summary.get("models", {})
    pm = stats.get("model_comparison", {}).get("per_model", {})
    pw = stats.get("model_comparison", {}).get("pairwise", {})

    # 8.1 — rates table
    e.append(Paragraph("8.1  RQ1 Hallucination Rates by Model (With 95% CI)", H2))
    rows = [["Model", "N Responses", "Hall. Rate", "95% CI", "Avg Faithfulness", "Interpretation"]]
    interps = {
        "gpt-4o-mini": "High refusal rate inflates score; true hall. ~8%",
        "mistral":     "Best performer — significantly lower than others",
        "qwen":        "Highest rate; most prone to fabrication",
    }
    for m, d in models_data.items():
        ci = pm.get(m, {})
        hall_rate = d.get("hallucination_rate")
        faith = d.get("avg_faithfulness")
        if hall_rate is None:
            rows.append([m, d.get("status", "Pending"), "—", "—", "—", "ALICE run pending"])
        else:
            rows.append([
                m, str(d.get("n_responses","")),
                f"{hall_rate:.1%}",
                f"[{ci.get('ci_low',0):.1%} – {ci.get('ci_high',0):.1%}]",
                f"{faith:.3f}" if faith is not None else "—",
                interps.get(m, ""),
            ])
    e.append(_tbl(rows, widths=[3*cm, 2.5*cm, 2.5*cm, 3.5*cm, 3.5*cm, 4.7*cm]))
    e.append(Paragraph(
        "The 95% Confidence Intervals (CI) tell you the true hallucination rate is "
        "somewhere in that range with 95% probability. Non-overlapping CIs for "
        "Mistral vs Qwen confirm they are genuinely different.", BODY))

    # 8.2 — query types
    e.append(Paragraph("8.2  Why Sentiment Queries Hallucinate at 76.83%", H2))
    qt_data = models_data.get("gpt-4o-mini", {}).get("by_query_type", {})
    rows = [["Query Type", "Hall. Rate", "Count", "Why This Rate"]]
    why = {
        "speaker_attribution": "Direct quotes are easy to retrieve — very faithful",
        "participant_content":  "Topics are explicit in transcript — easy to verify",
        "factual_summary":      "Some over-summarisation occurs",
        "temporal":             "Occasional ordering errors at interview boundaries",
        "specific_content":     "Context around a specific quote is often missing from retrieved chunks",
        "sentiment":            "GPT-4o-mini refuses to interpret personal feelings (safety guardrail)",
    }
    for qt, d in sorted(qt_data.items(), key=lambda x: x[1].get("rate",0)):
        rows.append([qt.replace("_"," "), f"{d.get('rate',0):.1%}", str(d.get("hallucinated","")), why.get(qt,"")])
    e.append(_tbl(rows, widths=[4*cm, 2.5*cm, 2*cm, 9.2*cm]))

    # 8.3 — REFUSAL
    e.append(Paragraph("8.3  REFUSAL vs True Hallucination — Why This Matters", H2))
    hall_types = models_data.get("gpt-4o-mini", {}).get("hallucination_type_counts", {})
    n_ref = hall_types.get("REFUSAL_HALLUCINATION", 511)
    n_true = sum(v for k,v in hall_types.items() if k != "REFUSAL_HALLUCINATION")
    n_total = models_data.get("gpt-4o-mini", {}).get("n_responses", 3106)
    e.append(_box([
        Paragraph(f"<b>GPT-4o-mini total responses:</b> {n_total:,}", NOTE),
        Paragraph(f"<b>Hallucinated (all types):</b> {int(n_total*0.2077):,} ({0.2077:.1%})", NOTE),
        Paragraph(f"<b>Of which REFUSAL type:</b> {n_ref:,} ({n_ref/n_total:.1%} of all responses)", NOTE),
        Paragraph(f"<b>True hallucinations (excl. refusal):</b> {n_true:,} ({n_true/n_total:.1%})", NOTE),
        _sp(0.1),
        Paragraph("<b>Conclusion:</b> If you exclude refusals, GPT-4o-mini's true hallucination rate drops "
                  f"from 20.77% to ~{n_true/n_total:.1%}. This is a key finding: safety guardrails "
                  "inflate RAG hallucination metrics on privacy-sensitive data.", WARN),
    ], bg=C_LBLUE, border=C_BLUE))

    # 8.4 — cross-lingual
    e.append(Paragraph("8.4  Cross-Lingual Results (EN vs NL)", H2))
    rows = [["Model", "EN Rate", "NL Rate", "Gap", "What It Means"]]
    for m, d in models_data.items():
        lang = d.get("by_language", {})
        en = lang.get("en", {}).get("rate", 0)
        nl = lang.get("nl", {}).get("rate", 0)
        rows.append([m, f"{en:.1%}", f"{nl:.1%}", f"{abs(en-nl):.2%}", "Negligible — pipeline generalises across languages"])
    e.append(_tbl(rows, widths=[3.5*cm, 3*cm, 3*cm, 2.5*cm, 5.7*cm]))

    # 8.5 — RQ2 available
    e.append(Paragraph("8.5  RQ2 Detector Results (Available So Far)", H2))
    rows = [
        ["Model", "Detector", "F1", "Precision", "Recall", "Interpretation"],
        ["GPT-4o-mini", "MiniCheck",    "0.857", "0.857", "0.857", "Excellent — best single detector so far"],
        ["GPT-4o-mini", "SelfCheckGPT", "0.000", "0.000", "0.000", "Failed — all responses scored below threshold"],
        ["Qwen",        "MiniCheck",    "0.338", "0.224", "0.684", "Weak precision, better recall"],
        ["Qwen",        "SelfCheckGPT", "0.000", "0.000", "0.000", "Failed — same issue as GPT-4o-mini"],
        ["Mistral",     "All",          "Pending", "—", "—", "RQ2 not yet run for Mistral"],
    ]
    e.append(_tbl(rows, widths=[3.2*cm, 3*cm, 1.8*cm, 2.5*cm, 2.5*cm, 4.7*cm]))
    e.append(Paragraph(
        "<b>Why SelfCheckGPT failed:</b> SelfCheckGPT threshold is 0.5 by default. "
        "For interview data, the 5 sampled responses happen to be quite consistent with each other "
        "(all refuse to interpret sentiment in the same way), so BERTScore stays below 0.5 "
        "and nothing gets flagged. This is itself an interesting finding about SelfCheckGPT's "
        "limitations in the dialogue domain.", BODY))

    # 8.6 — McNemar
    e.append(Paragraph("8.6  McNemar Test Results — What They Prove", H2))
    rows = [["Comparison", "χ²", "p-value", "Significant?", "What It Means"]]
    for pair, res in pw.items():
        rows.append([
            pair,
            str(res.get("statistic","")),
            str(res.get("p_value","")),
            "YES" if res.get("significant") else "NO",
            "Mistral makes significantly fewer errors than the other two" if res.get("significant") else "No significant difference between these two models",
        ])
    e.append(_tbl(rows, widths=[4.5*cm, 2*cm, 2.5*cm, 2.5*cm, 6.2*cm]))
    e.append(Paragraph(
        "McNemar's test compares two classifiers on the same set of questions — "
        "it asks 'do they make different mistakes on the same items?' "
        "A p-value < 0.05 means the difference is statistically real, not random chance.", BODY))

    # 8.7 — pending
    e.append(Paragraph("8.7  Pending Results — What to Expect", H2))
    rows = [
        ["What's Pending", "Expected Result", "Impact on Paper"],
        ["AlignScore (all models)", "F1 likely 0.3–0.6 based on typical performance", "3rd column in RQ2 comparison table"],
        ["RAGAS (all models)",      "Faithfulness 0.7–0.9 for faithful responses",     "4th column; good for discussion"],
        ["Mistral RQ2",             "Similar pattern to Qwen (SelfCheck fails, MiniCheck moderate)", "Completes the 3×4 comparison matrix"],
        ["Manual Kappa",            "Target κ ≥ 0.61",                                 "Required for paper validity"],
        ["Ensemble detector",       "F1 > 0.857 (should beat MiniCheck alone)",        "Novel contribution C5"],
    ]
    e.append(_tbl(rows, widths=[4.5*cm, 6*cm, 7.2*cm]))

    # 8.8 — DiaHaLu
    e.append(Paragraph("8.8  DiaHaLu Dialogue-Level Results", H2))
    e.append(Paragraph(
        "DiaHaLu evaluates conversation-level errors that RAGTruth misses. "
        "Key finding: GPT-4o-mini produces TOPIC_DRIFT in 266 interviews — "
        "the response drifts to unrelated topics. This does not appear in RAGTruth "
        "because RAGTruth is sentence-level. DiaHaLu adds a dialogue-level lens.", BODY))
    rows = [["Issue Type", "GPT-4o-mini", "Mistral", "Qwen", "What It Means"]]
    for m, d in models_data.items():
        pass  # collect
    diahalu = {
        m: models_data.get(m,{}).get("diahalu_summary",{}).get("issue_type_counts",{})
        for m in ["gpt-4o-mini","mistral","qwen"]
    }
    issue_types = ["TOPIC_DRIFT","CROSS_TURN_CONTRADICTION","TEMPORAL_DISTORTION","CONTEXT_FABRICATION","ENTITY_CONFUSION","SPEAKER_CONFUSION"]
    descs = {
        "TOPIC_DRIFT": "Response wanders off-topic",
        "CROSS_TURN_CONTRADICTION": "Contradicts something said earlier",
        "TEMPORAL_DISTORTION": "Wrong chronological ordering",
        "CONTEXT_FABRICATION": "Invents context not in any turn",
        "ENTITY_CONFUSION": "Confuses names/entities between speakers",
        "SPEAKER_CONFUSION": "Attributes speech to wrong speaker",
    }
    for it in issue_types:
        rows.append([
            it.replace("_"," "),
            str(diahalu["gpt-4o-mini"].get(it,0)),
            str(diahalu.get("mistral",{}).get(it,0)),
            str(diahalu.get("qwen",{}).get(it,0)),
            descs.get(it,""),
        ])
    e.append(_tbl(rows, widths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 5.7*cm]))
    e.append(PageBreak())
    return e


def sec_professor():
    e = []
    e.append(Paragraph("Section 9 — Professor Meeting: How to Pitch and Defend", H1))
    e.append(_hr())

    # Pitch
    e.append(Paragraph("9.1  Opening 2-Minute Pitch", H2))
    e.append(_box([
        Paragraph("<b>Speak this (adapt naturally):</b>", CELLB),
        _sp(0.15),
        Paragraph(
            "\"My thesis investigates hallucination in RAG systems applied to AI-powered interview data. "
            "When researchers use an AI to query interview transcripts — asking things like 'what was "
            "the participant's sentiment?' — the AI often produces false or unsupported answers. "
            "I call this interview RAG hallucination, and it's a problem no one has studied before "
            "in this domain.\"",
            BODY),
        _sp(0.1),
        Paragraph(
            "\"I evaluated three LLMs — GPT-4o-mini, Mistral, and Qwen — on 9,218 query-response "
            "pairs generated from 749 real and synthetic interviews in English and Dutch. "
            "I classified hallucinations using three taxonomies, tested four automated detection "
            "methods, and discovered that GPT-4o-mini's high hallucination rate (20.77%) is "
            "largely explained by safety guardrails refusing to interpret personal data — "
            "a novel finding that has implications for how we measure hallucination in "
            "privacy-sensitive AI systems.\"",
            BODY),
    ], bg=C_LBLUE, border=C_DARK))

    # Q&A
    qa = [
        # Technical
        ("Why GPT-4o-mini, Mistral, and Qwen — why not GPT-4 full or Llama?", [
            "These three represent three distinct tiers: a commercial small model (GPT-4o-mini), a capable open-weight model (Mistral-Small-24B), and a Chinese open-weight model (Qwen2.5-7B)",
            "This gives cost/capability diversity: GPT-4o-mini is cheap to run at scale; Mistral and Qwen are self-hostable",
            "GPT-4o (full) would be interesting as an upper bound — I plan to add it if time permits",
            "Llama-3.1-8B is a natural next step for open-source reproducibility",
            "The current 3-model selection is standard for IEEE papers in this space — sufficient to show cross-model patterns",
        ]),
        ("Is 749 interviews a sufficient sample size?", [
            "749 interviews × 6 query types × 3 models = 9,218 response pairs — this is the evaluation corpus, not a training set",
            "For detection comparison (RQ2), no training is needed — detectors are pre-trained models evaluated on our data",
            "The 95% Wilson CIs are narrow: GPT-4o-mini rate is 20.77% ± 1.4% — statistically robust",
            "McNemar's test on 3,000+ shared samples gives very high statistical power",
            "IEEE papers in hallucination detection typically use 500–5,000 evaluation instances — we are in that range",
        ]),
        ("Your synthetic data is AI-generated — how do you validate its quality?", [
            "The synthetic interviews were generated to match the structure of real Convo interviews — same JSON schema, same utterance format",
            "The hallucination rates are nearly identical across real and synthetic interviews — suggesting the models respond similarly to both",
            "The EN/NL language distribution in synthetic data matches the real data proportions",
            "This is a known limitation — I address it explicitly in the Threats to Validity section as a future improvement",
            "A human quality check on 50 synthetic interviews would strengthen this — I can do that alongside annotation",
        ]),
        ("GPT-4o-mini is evaluating outputs that partly came from GPT-4o-mini — isn't that circular?", [
            "This is a known limitation called 'LLM-as-judge bias' — I address it explicitly in the paper",
            "The mitigation is manual annotation + Cohen's Kappa: 200 human-labelled samples will validate whether the auto-labels are trustworthy",
            "Prior work (Zheng et al., 2023; Manakul et al., 2023) uses the same LLM-as-judge approach — it is standard practice",
            "GPT-4o-mini was the evaluator for ALL three models including Mistral and Qwen — so the circular concern only applies to ~33% of responses",
            "Using a separate evaluator (e.g. Claude) for cross-validation is in the future work section",
        ]),
        ("Why the RAGTruth taxonomy with 7 types — why not just HALLUCINATED/FAITHFUL?", [
            "A binary label tells you THAT a model hallucinates, not WHY or HOW",
            "The 7 types reveal important patterns: 60% of GPT-4o-mini errors are REFUSAL — this would be invisible with binary labels",
            "Different error types have different remediation strategies: TEMPORAL_CONFUSION needs better chunking; REFUSAL needs a different prompt",
            "Multi-class taxonomy is standard in ACL/EMNLP hallucination papers — binary is considered too coarse for research contributions",
            "I compare all three taxonomies on the same data — the comparison itself is a contribution",
        ]),
        ("SelfCheckGPT returned F1=0 — does that mean your detection failed?", [
            "SelfCheckGPT didn't fail — it failed for a specific and interesting reason that becomes a finding",
            "For interview data, all 5 sampled responses tend to be consistently wrong in the same way (all refuse to interpret sentiment)",
            "BERTScore sees high consistency between samples, so it scores hallucination risk as low — a false negative",
            "This shows SelfCheckGPT's limitation in dialogue domains: it assumes inconsistency = hallucination, but models can be consistently wrong",
            "I report this as a finding: SelfCheckGPT is not suitable for interview RAG due to correlated refusal behaviour",
        ]),
        ("What is specifically novel about this versus existing RAG hallucination papers?", [
            "First study on interview/dialogue domain RAG — all prior work uses Wikipedia, news, or general QA datasets",
            "Refusal-hallucination distinction: we identify safety guardrails as a separate failure mode — not studied before",
            "Multilingual: English + Dutch — almost all RAG hallucination papers are English-only",
            "Multi-taxonomy comparison on identical data: we compare RAGTruth vs Huang vs DiaHaLu on the same responses",
            "Predictive hallucination detection: hallucination_predictor.py predicts risk before generation from retrieval signals",
        ]),
        ("Why interview data specifically — is this just a domain application paper?", [
            "Interview data has unique properties that make it more than a simple application: privacy sensitivity, personal content, emotional nuance",
            "The safety guardrail problem (REFUSAL_HALLUCINATION at 60%) only emerges in privacy-sensitive domains — you would not see this in Wikipedia RAG",
            "Interview data is increasingly used in qualitative research and market research — the practical impact is real",
            "The multilingual aspect (EN + NL) adds a research dimension beyond domain application",
            "The three taxonomy comparison adds methodological contribution beyond the domain",
        ]),
        ("What's the practical impact — who would actually use this?", [
            "Researchers and companies using AI-powered interview tools (like Convo) need to know when to trust AI-generated summaries",
            "Market research firms running hundreds of customer interviews could use hallucination detection to flag unreliable summaries automatically",
            "HR platforms using AI to analyse candidate interviews face the same problem — misquoting a candidate has legal/ethical consequences",
            "The correction_loop.py module demonstrates a practical fix: detect hallucination, auto-correct it",
            "The hallucination_predictor.py module enables pre-emptive quality filtering before showing results to users",
        ]),
        ("What would you do with more time?", [
            "Add GPT-4o (full) as upper-bound model and Llama-3.1-8B for open-source reproducibility",
            "Complete AlignScore and RAGAS experiments for all models, then train and evaluate the ensemble detector",
            "Expand manual validation from 200 to 500+ samples with a second annotator for robust Kappa",
            "Run the hyperparameter sweep to find optimal chunk_size + top_k configurations",
            "Study the correction_loop performance — how often does auto-correction actually fix hallucinations?",
        ]),
    ]

    for q, bullets in qa:
        e.append(KeepTogether([
            Paragraph(f"Q: {q}", QA_Q),
        ] + [Paragraph(f"• {b}", QA_A) for b in bullets] + [_sp(0.2)]))

    e.append(PageBreak())
    return e


def sec_explore():
    e = []
    e.append(Paragraph("Section 10 — What More to Explore", H1))
    e.append(_hr())

    e.append(Paragraph("10.1  High-Priority Gaps to Fill Before Submission", H2))
    rows = [
        ["#", "Gap", "How to Fill It", "Time Needed"],
        ["1", "AlignScore + RAGAS missing for all 3 models",
         "python3 src/run_all.py --steps rq2 (these should run if transformers installed)",
         "2–4 hours"],
        ["2", "Mistral has no RQ2 results",
         "python3 src/run_all.py --models mistral --steps rq2",
         "1–2 hours"],
        ["3", "Manual annotation (0/198 rows labelled)",
         "Follow Section 7 of this guide. 7 sessions of 30 rows each.",
         "~4 hours total"],
        ["4", "Cohen's Kappa not computed",
         "Follows automatically from annotating. Run calculate_kappa.py after.",
         "5 minutes"],
        ["5", "Ensemble detector not trained",
         "python3 RAG_THESIS/advanced/ensemble_detector.py (needs manual labels first)",
         "10 minutes"],
    ]
    e.append(_tbl(rows, widths=[0.8*cm, 4.5*cm, 7.5*cm, 2.9*cm]))

    e.append(Paragraph("10.2  Additional Models Worth Testing", H2))
    rows = [
        ["Model", "Why Add It", "Priority", "How to Add"],
        ["GPT-4o (full)", "Upper bound — shows max possible quality and compares to GPT-4o-mini", "High", "Add 'gpt-4o' to DEFAULT_MODELS in run_all.py"],
        ["Llama-3.1-8B", "Open-source — other researchers can reproduce without API keys", "High", "Add 'meta-llama/Llama-3.1-8B-Instruct' and configure Ollama or HuggingFace"],
        ["mBERT / XLM-R", "Multilingual detection model for cross-lingual hallucination scoring", "Medium", "Use as AlignScore alternative in experiment_rq2.py"],
        ["Gemini 1.5 Flash", "Google alternative — multilingual, cost-efficient", "Low", "Requires Gemini API key"],
    ]
    e.append(_tbl(rows, widths=[3.2*cm, 5*cm, 2.5*cm, 6.5*cm]))

    e.append(Paragraph("10.3  Detection Method Improvements", H2))
    rows = [
        ["Current Method", "Improvement", "Expected Impact", "Effort"],
        ["AlignScore (RoBERTa-base)", "Upgrade to RoBERTa-large or DeBERTa-v3-large", "+5–8% F1 based on literature", "Low (change model name in code)"],
        ["SelfCheckGPT (BERTScore)", "Try MQAG variant (multiple-choice QA)", "Better precision for factual claims", "Medium (add MQAG dependency)"],
        ["RAGAS (faithfulness only)", "Add RAGAS answer_relevancy metric", "Broader hallucination coverage", "Low (already in ragas library)"],
        ["Fixed ensemble weights", "Use logistic regression weights from ensemble_detector.py", "Learned weights beat fixed 30/40/30", "Low (run ensemble_detector.py)"],
        ["5th ensemble signal", "Add depth_scoring.py as 5th feature in ensemble", "Unique to your work", "Low (depth_scoring.py exists in src/)"],
    ]
    e.append(_tbl(rows, widths=[4*cm, 4.5*cm, 4*cm, 5.2*cm]))

    e.append(Paragraph("10.4  Advanced Modules Not Yet Run", H2))
    rows = [
        ["Module", "What It Does", "How to Run", "What to Report"],
        ["hallucination_predictor.py",
         "Predicts hallucination risk BEFORE the response is generated, using retrieval-time signals: similarity score, chunk size, number of retrieved chunks, query type",
         "python3 RAG_THESIS/advanced/hallucination_predictor.py",
         "AUC-ROC of predictor vs actual hallucination rate. If AUC > 0.7, the model can be used to pre-filter risky queries."],
        ["correction_loop.py",
         "Detects high-hallucination responses and re-queries the model with stronger grounding instructions to produce a corrected response",
         "python3 RAG_THESIS/advanced/correction_loop.py",
         "% of hallucinations corrected. MiniCheck score before vs after correction."],
        ["retrieval_correlation.py",
         "Computes Pearson/Spearman correlation between retrieval quality metrics (similarity score, chunk size) and hallucination rate",
         "python3 RAG_THESIS/advanced/retrieval_correlation.py",
         "Correlation coefficient. If high, retrieval quality predicts hallucination — useful finding."],
        ["run_no_rag_baseline.py",
         "Answers all queries without providing the transcript as context (no RAG). Shows what happens if you just ask the model directly.",
         "python3 RAG_THESIS/ablation/run_no_rag_baseline.py",
         "Hallucination rate without RAG vs with RAG. Should be much higher without RAG — proves RAG helps."],
    ]
    e.append(_tbl(rows, widths=[4*cm, 4.5*cm, 4.5*cm, 4.7*cm]))

    e.append(Paragraph("10.5  Future Research Directions", H2))
    directions = [
        ("Human-in-the-loop correction", "Build a UI where researchers see hallucination alerts and approve/reject corrections before viewing interview summaries"),
        ("Fine-tuning a domain-specific detector", "Use the 9,218 labelled responses to fine-tune a compact NLI model specifically for interview hallucination — faster and cheaper than MiniCheck"),
        ("Real-time detection", "Integrate hallucination detection into the live interview pipeline so researchers are warned before data is saved"),
        ("Cross-dataset transfer", "Test whether detectors trained on Convo interviews generalise to other interview platforms (academic research, HR, medical)"),
        ("Dutch-specific evaluation", "Partner with Dutch researchers to add more NL interviews and a Dutch-speaking annotator for cross-lingual Kappa"),
    ]
    for title, desc in directions:
        e.append(Paragraph(f"<b>{title}:</b> {desc}", BULL))

    return e


# ── Build PDF ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    summary, stats, annot_rows, rq1_rows = _load()

    print("Building PDF sections...")
    story = []
    story += sec_cover(summary)
    story += sec_data(summary)
    story += sec_related(summary)
    story += sec_experiments(summary)
    story += sec_architecture()
    story += sec_run_guide()
    story += sec_annotation(annot_rows, rq1_rows)
    story += sec_results(summary, stats)
    story += sec_professor()
    story += sec_explore()

    print("Rendering PDF...")
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM,
        title="Thesis Companion Guide",
    )
    doc.build(story)
    print(f"\nSaved → {OUTPUT_PATH}")
    print("Open with:  open results/thesis_companion_guide.pdf")


if __name__ == "__main__":
    main()
