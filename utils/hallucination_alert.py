"""
Hallucination Alert System
============================
Combines all 3 RQ2 detection methods into a unified alert that pinpoints
the EXACT location (character offsets) of hallucinations in a RAG response.

Detection methods:
  SelfCheckGPT  — sentence-level self-consistency (BERTScore)
  MiniCheck     — claim-level NLI verification against transcript
  RAGAS         — statement-level faithfulness verification (Es et al., EACL 2024)

Output:
  .to_json()          → dict with span list + per-method scores
  .to_pdf(path)       → single-response PDF alert report (reportlab)
  .to_batch_pdf(...)  → multi-response combined PDF

Usage (standalone):
    python hallucination_alert.py \\
        --response "The participant said they love their team." \\
        --context  "Participant: I actually find it hard to work with my colleagues." \\
        --query    "How does the participant feel about their team?" \\
        --out      results/alert_example.pdf
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

# ------------------------------------------------------------------ sentence splitting

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if len(p.split()) >= 3]


def _char_offsets(full_text: str, sentence: str) -> tuple:
    """Return (start, end) char offsets of sentence in full_text."""
    idx = full_text.find(sentence)
    if idx == -1:
        return (-1, -1)
    return (idx, idx + len(sentence))


# ------------------------------------------------------------------ method runners

def _run_selfcheck(response: str, context: str, query: str,
                   sampled_responses: List[str]) -> List[Dict]:
    """
    Returns list of {sentence, char_start, char_end, selfcheck_score}.
    Score 0=faithful, 1=hallucinated.
    """
    sentences = _split_sentences(response)
    if not sentences or not sampled_responses:
        return []

    scores = []

    # Try selfcheckgpt library first
    try:
        from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
        sc = SelfCheckBERTScore()
        raw_scores = sc.predict(sentences=sentences, sampled_passages=sampled_responses)
        scores = [float(s) for s in raw_scores]
    except Exception:
        # Fall back to bert_score
        try:
            from bert_score import score as bert_score_fn
            for sent in sentences:
                refs = sampled_responses
                cands = [sent] * len(refs)
                _, _, F1 = bert_score_fn(cands, refs, lang="en", verbose=False)
                scores.append(1.0 - float(F1.mean()))
        except Exception:
            # Simple word overlap
            for sent in sentences:
                words = set(sent.lower().split())
                overlaps = []
                for s in sampled_responses:
                    sw = set(s.lower().split())
                    overlaps.append(len(words & sw) / len(words) if words else 0)
                scores.append(1.0 - (sum(overlaps) / len(overlaps) if overlaps else 0))

    results = []
    for sent, score in zip(sentences, scores):
        start, end = _char_offsets(response, sent)
        results.append({
            "sentence": sent,
            "char_start": start,
            "char_end": end,
            "selfcheck_score": round(score, 4),
        })
    return results


def _run_minicheck(response: str, context: str) -> List[Dict]:
    """
    Returns list of {claim, char_start, char_end, is_supported, minicheck_score}.
    """
    claims_raw = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response.strip())
                  if len(s.split()) >= 4]
    if not claims_raw:
        return []

    claim_results = []

    # Try MiniCheck library
    USE_MINICHECK = False
    USE_HF = False
    nli_pipe = None

    try:
        from minicheck.minicheck import MiniCheck
        scorer = MiniCheck(model_name="flan-t5-large", cache_dir="./minicheck_cache")
        USE_MINICHECK = True
    except Exception:
        try:
            from transformers import pipeline as hf_pipeline
            nli_pipe = hf_pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,
            )
            USE_HF = True
        except Exception:
            pass

    for claim in claims_raw:
        start, end = _char_offsets(response, claim)
        try:
            if USE_MINICHECK:
                pred_label, pred_prob, _, _ = scorer.score(
                    docs=[context[:3000]], claims=[claim]
                )
                is_supported = pred_label[0] == 1
                confidence = float(pred_prob[0])
            elif USE_HF and nli_pipe:
                inp = f"{context[:1500]} [SEP] {claim}"
                res = nli_pipe(inp, truncation=True)
                is_supported = "entail" in res[0]["label"].lower()
                confidence = res[0]["score"]
            else:
                # Simple overlap heuristic
                claim_words = set(claim.lower().split())
                ctx_words = set(context.lower().split())
                overlap = len(claim_words & ctx_words) / len(claim_words) if claim_words else 0
                is_supported = overlap > 0.3
                confidence = overlap

            claim_results.append({
                "claim": claim,
                "char_start": start,
                "char_end": end,
                "is_supported": is_supported,
                "minicheck_score": round(1.0 - confidence if is_supported else confidence, 4),
            })
        except Exception as e:
            claim_results.append({
                "claim": claim,
                "char_start": start,
                "char_end": end,
                "is_supported": None,
                "minicheck_score": 0.5,
                "error": str(e),
            })

    return claim_results


def _run_ragas_faithfulness(response: str, context: str, query: str) -> List[Dict]:
    """
    Returns list of {statement, char_start, char_end, is_supported, confidence}.
    Uses RAGAS faithfulness metric (Es et al., EACL 2024).
    Each entry is one sentence of the response with its supported/unsupported verdict.
    Falls back to NLI-DeBERTa sentence checking if RAGAS/OpenAI is unavailable.
    """
    sentences = _split_sentences(response)
    if not sentences:
        return []

    results = []

    # ── Primary: RAGAS faithfulness (paper: Es et al., EACL 2024) ─────────
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import faithfulness as ragas_faithfulness
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI
            ragas_faithfulness.llm = LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )
        except Exception:
            pass  # use RAGAS default LLM

        sample = SingleTurnSample(
            user_input=query or "Summarize the key points.",
            response=response,
            retrieved_contexts=[context[:3000]],
        )
        dataset = EvaluationDataset(samples=[sample])
        result_ragas = evaluate(dataset, metrics=[ragas_faithfulness])
        overall_score = float(result_ragas["faithfulness"])

        # Use sentence-level NLI for per-sentence granularity alongside RAGAS score
        nli_pipe = None
        try:
            from transformers import pipeline as hf_pipeline
            nli_pipe = hf_pipeline("text-classification",
                                   model="cross-encoder/nli-deberta-v3-small", device=-1)
        except Exception:
            pass

        for sent in sentences:
            start, end = _char_offsets(response, sent)
            if nli_pipe:
                inp = f"{context[:2000]} [SEP] {sent}"
                res = nli_pipe(inp, truncation=True)[0]
                is_sup = "entail" in res["label"].lower()
                sent_conf = res["score"] if is_sup else 1.0 - res["score"]
                # Blend RAGAS overall with per-sentence NLI
                blended = round((overall_score + sent_conf) / 2, 3)
                results.append({
                    "statement": sent,
                    "char_start": start, "char_end": end,
                    "is_supported": is_sup,
                    "confidence": blended,
                })
            else:
                results.append({
                    "statement": sent,
                    "char_start": start, "char_end": end,
                    "is_supported": overall_score >= 0.5,
                    "confidence": round(overall_score, 3),
                })
        return results

    except Exception:
        pass

    # ── Fallback: NLI sentence-level only ─────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline
        nli = hf_pipeline("text-classification",
                           model="cross-encoder/nli-deberta-v3-small", device=-1)
        for sent in sentences:
            start, end = _char_offsets(response, sent)
            inp = f"{context[:2000]} [SEP] {sent}"
            res = nli(inp, truncation=True)[0]
            is_sup = "entail" in res["label"].lower()
            conf = res["score"] if is_sup else 1.0 - res["score"]
            results.append({
                "statement": sent,
                "char_start": start, "char_end": end,
                "is_supported": is_sup,
                "confidence": round(conf, 3),
            })
    except Exception:
        pass

    return results


# ------------------------------------------------------------------ combined alert

def _combined_risk(sc_score: float, mc_score: float, ld_conf: float) -> float:
    """Weighted combination: selfcheck 30%, minicheck 40%, ragas 30%."""
    return round(0.3 * sc_score + 0.4 * mc_score + 0.3 * ld_conf, 4)


class HallucinationAlert:
    """
    Unified hallucination alert for a single RAG response.

    Parameters
    ----------
    response         : the RAG-generated answer text
    context          : the interview transcript used as RAG context
    query            : the question that prompted the response
    sampled_responses: 5 stochastic samples (for SelfCheckGPT); can be empty list
    """

    def __init__(self, response: str, context: str, query: str,
                 sampled_responses: Optional[List[str]] = None):
        self.response = response
        self.context = context
        self.query = query
        self.sampled_responses = sampled_responses or []

        t0 = time.time()
        self._sc    = _run_selfcheck(response, context, query, self.sampled_responses)
        self._mc    = _run_minicheck(response, context)
        self._ragas = _run_ragas_faithfulness(response, context, query)
        self.latency = round(time.time() - t0, 2)

        self.spans = self._merge_spans()

    # ---- internal: merge three method outputs into unified span list -----------

    def _merge_spans(self) -> List[Dict]:
        """
        Build a unified list of flagged spans by overlapping the three methods.
        Each span: {text, char_start, char_end, combined_risk, methods, suggested_type}
        """
        # Collect candidate spans from all methods
        candidates = []

        for item in self._sc:
            if item["selfcheck_score"] > 0.5:
                candidates.append({
                    "char_start": item["char_start"],
                    "char_end": item["char_end"],
                    "text": item["sentence"],
                    "sc_score": item["selfcheck_score"],
                    "mc_score": 0.0,
                    "ld_conf": 0.0,
                    "methods": ["selfcheck"],
                })

        for item in self._mc:
            if item.get("is_supported") is False:
                candidates.append({
                    "char_start": item["char_start"],
                    "char_end": item["char_end"],
                    "text": item["claim"],
                    "sc_score": 0.0,
                    "mc_score": item.get("minicheck_score", 0.7),
                    "ld_conf": 0.0,
                    "methods": ["minicheck"],
                })

        for item in self._ragas:
            if not item.get("is_supported", True):
                candidates.append({
                    "char_start": item["char_start"],
                    "char_end": item["char_end"],
                    "text": item["statement"],
                    "sc_score": 0.0,
                    "mc_score": 0.0,
                    "ld_conf": item.get("confidence", 0.0),
                    "methods": ["ragas"],
                })

        if not candidates:
            return []

        # Merge overlapping candidates (greedy)
        candidates.sort(key=lambda x: x["char_start"])
        merged = []
        for c in candidates:
            if merged and c["char_start"] <= merged[-1]["char_end"] and c["char_start"] >= 0:
                prev = merged[-1]
                # Expand and combine scores
                prev["char_end"] = max(prev["char_end"], c["char_end"])
                prev["sc_score"]  = max(prev["sc_score"],  c["sc_score"])
                prev["mc_score"]  = max(prev["mc_score"],  c["mc_score"])
                prev["ld_conf"]   = max(prev["ld_conf"],   c["ld_conf"])
                for m in c["methods"]:
                    if m not in prev["methods"]:
                        prev["methods"].append(m)
            else:
                merged.append(dict(c))

        # Compute combined risk and suggest type
        result = []
        for s in merged:
            risk = _combined_risk(s["sc_score"], s["mc_score"], s["ld_conf"])
            if risk < 0.3:
                continue  # below threshold, skip

            # Heuristic type suggestion
            text = s["text"].lower()
            if any(w in text for w in ["not available", "not in the transcript", "cannot find"]):
                suggested_type = "REFUSAL_HALLUCINATION"
            elif s["ld_conf"] > 0.5 and s["mc_score"] > 0.5:
                suggested_type = "EVIDENT_CONFLICT"
            elif s["mc_score"] > 0.6:
                suggested_type = "BASELESS_INFO"
            elif s["sc_score"] > 0.6:
                suggested_type = "SUBTLE_CONFLICT"
            else:
                suggested_type = "UNCERTAIN"

            result.append({
                "text": s["text"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
                "combined_risk": risk,
                "methods": s["methods"],
                "method_scores": {
                    "selfcheck": round(s["sc_score"], 4),
                    "minicheck": round(s["mc_score"], 4),
                    "ragas": round(s["ld_conf"], 4),
                },
                "suggested_type": suggested_type,
            })

        return result

    # ---- public outputs -------------------------------------------------------

    def to_json(self) -> Dict:
        """Return full alert as a JSON-serialisable dict."""
        return {
            "response": self.response,
            "query": self.query,
            "overall_risk": round(sum(s["combined_risk"] for s in self.spans) / max(len(self.spans), 1), 4),
            "num_flagged_spans": len(self.spans),
            "is_hallucinated": len(self.spans) > 0,
            "latency_s": self.latency,
            "flagged_spans": self.spans,
            "method_detail": {
                "selfcheck": self._sc,
                "minicheck": self._mc,
                "ragas": self._ragas,
            },
        }

    def to_pdf(self, output_path: str):
        """Generate a PDF alert report for this single response."""
        _generate_alert_pdf([self.to_json()], Path(output_path))
        print(f"Alert PDF saved: {output_path}")

    @staticmethod
    def to_batch_pdf(alerts: List["HallucinationAlert"], output_path: str):
        """Generate one combined PDF for multiple alerts."""
        data = [a.to_json() for a in alerts]
        _generate_alert_pdf(data, Path(output_path))
        print(f"Batch alert PDF saved: {output_path}")


# ------------------------------------------------------------------ PDF generation

def _risk_color(risk: float):
    """Return (R, G, B) tuple scaled 0-1 based on risk."""
    if risk >= 0.7:
        return (0.85, 0.15, 0.15)   # red
    elif risk >= 0.4:
        return (0.95, 0.55, 0.0)    # orange
    else:
        return (0.1, 0.65, 0.1)     # green


def _generate_alert_pdf(alerts_data: List[Dict], output_path: Path):
    """Render alert data to a PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        print("ERROR: pip install reportlab")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle("title",   fontSize=16, spaceAfter=6,  fontName="Helvetica-Bold")
    heading_style = ParagraphStyle("heading", fontSize=12, spaceAfter=4,  fontName="Helvetica-Bold", textColor=colors.HexColor("#2c3e50"))
    body_style    = ParagraphStyle("body",    fontSize=9,  spaceAfter=4,  fontName="Helvetica", leading=13)
    code_style    = ParagraphStyle("code",    fontSize=8,  spaceAfter=3,  fontName="Courier",   leading=11, backColor=colors.HexColor("#f4f4f4"))
    small_style   = ParagraphStyle("small",   fontSize=8,  spaceAfter=2,  fontName="Helvetica", textColor=colors.grey)

    story = []

    # ---- Cover heading
    story.append(Paragraph("Hallucination Alert Report", title_style))
    story.append(Paragraph(f"Total responses analysed: {len(alerts_data)}", small_style))
    hallucinated = sum(1 for a in alerts_data if a["is_hallucinated"])
    story.append(Paragraph(f"Responses with hallucinations detected: {hallucinated} / {len(alerts_data)}", small_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2c3e50")))
    story.append(Spacer(1, 0.3*cm))

    for idx, alert in enumerate(alerts_data, 1):
        story.append(Paragraph(f"Response #{idx}", heading_style))

        # Query
        story.append(Paragraph(f"<b>Query:</b> {alert.get('query', 'N/A')}", body_style))

        # Overall status
        risk_label = "HALLUCINATED" if alert["is_hallucinated"] else "FAITHFUL"
        risk_col_hex = "#c0392b" if alert["is_hallucinated"] else "#27ae60"
        story.append(Paragraph(
            f"<b>Status:</b> <font color='{risk_col_hex}'>{risk_label}</font>  "
            f"| Overall risk: {alert['overall_risk']:.2f}  "
            f"| Flagged spans: {alert['num_flagged_spans']}  "
            f"| Detection time: {alert['latency_s']}s",
            body_style
        ))
        story.append(Spacer(1, 0.2*cm))

        # ---- Page 1 content: annotated response text
        story.append(Paragraph("<b>Annotated Response:</b>", body_style))

        response_text = alert["response"]
        spans = sorted(alert["flagged_spans"], key=lambda x: x["char_start"])

        # Build annotated text with inline span markers
        if not spans:
            story.append(Paragraph(response_text or "(empty)", code_style))
        else:
            annotated_parts = []
            cursor = 0
            for span in spans:
                s, e = span["char_start"], span["char_end"]
                if s < 0:
                    continue
                if cursor < s:
                    safe = response_text[cursor:s].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    annotated_parts.append(safe)
                # Flagged span
                r, g, b = _risk_color(span["combined_risk"])
                hex_col = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                safe_span = response_text[s:e].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                annotated_parts.append(
                    f'<font color="{hex_col}"><b>[{safe_span}]</b></font>'
                )
                cursor = e
            if cursor < len(response_text):
                tail = response_text[cursor:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                annotated_parts.append(tail)

            full_annotated = "".join(annotated_parts)
            story.append(Paragraph(full_annotated, code_style))

        story.append(Spacer(1, 0.3*cm))

        # ---- Page 2 content: span breakdown table
        if spans:
            story.append(Paragraph("<b>Flagged Span Breakdown:</b>", body_style))
            table_data = [["#", "Flagged Text", "Risk", "Methods", "Type"]]
            for i, span in enumerate(spans, 1):
                methods_str = ", ".join(span["methods"])
                truncated = span["text"][:55] + "..." if len(span["text"]) > 55 else span["text"]
                table_data.append([
                    str(i),
                    truncated,
                    f"{span['combined_risk']:.2f}",
                    methods_str,
                    span["suggested_type"],
                ])

            col_widths = [0.5*cm, 9*cm, 1.2*cm, 3.5*cm, 3.8*cm]
            tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
            r_c, g_c, b_c = _risk_color(0.7)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ("VALIGN",     (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP",   (1, 1), (1, -1), True),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3*cm))

        # ---- Page 3 content: method agreement matrix
        story.append(Paragraph("<b>Method Agreement:</b>", body_style))
        sc_count  = sum(1 for s in spans if "selfcheck" in s["methods"])
        mc_count  = sum(1 for s in spans if "minicheck" in s["methods"])
        ld_count  = sum(1 for s in spans if "ragas"     in s["methods"])
        all_count = sum(1 for s in spans if len(s["methods"]) == 3)
        two_count = sum(1 for s in spans if len(s["methods"]) == 2)
        one_count = sum(1 for s in spans if len(s["methods"]) == 1)

        mat_data = [
            ["Method", "Flagged Spans", "Avg Score"],
            ["SelfCheckGPT (BERTScore)",
             str(sc_count),
             f"{sum(s['method_scores']['selfcheck'] for s in spans)/max(len(spans),1):.3f}"],
            ["MiniCheck (NLI Claim)",
             str(mc_count),
             f"{sum(s['method_scores']['minicheck'] for s in spans)/max(len(spans),1):.3f}"],
            ["RAGAS Faithfulness (EACL 2024)",
             str(ld_count),
             f"{sum(s['method_scores']['ragas'] for s in spans)/max(len(spans),1):.3f}"],
            ["All 3 methods agree", str(all_count), "—"],
            ["2 methods agree",     str(two_count),  "—"],
            ["1 method only",       str(one_count),  "—"],
        ]
        mat_tbl = Table(mat_data, colWidths=[8*cm, 3*cm, 3*cm], repeatRows=1)
        mat_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(mat_tbl)

        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.3*cm))

        # Page break between responses (not after last)
        if idx < len(alerts_data):
            story.append(PageBreak())

    doc.build(story)


# ------------------------------------------------------------------ CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hallucination alert on a single response")
    parser.add_argument("--response", type=str, required=True, help="The RAG response text")
    parser.add_argument("--context",  type=str, required=True, help="The interview transcript context")
    parser.add_argument("--query",    type=str, default="",    help="The query that prompted the response")
    parser.add_argument("--out",      type=str, default="results/alert_single.pdf", help="Output PDF path")
    args = parser.parse_args()

    print("Running hallucination alert...")
    alert = HallucinationAlert(
        response=args.response,
        context=args.context,
        query=args.query,
    )

    result = alert.to_json()
    print(f"\nStatus: {'HALLUCINATED' if result['is_hallucinated'] else 'FAITHFUL'}")
    print(f"Flagged spans: {result['num_flagged_spans']}")
    for span in result["flagged_spans"]:
        print(f"  [{span['combined_risk']:.2f}] ({span['suggested_type']}) \"{span['text'][:70]}\"")

    alert.to_pdf(args.out)
