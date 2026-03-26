"""
RQ1 Experiments: Hallucination Taxonomy
========================================
Three experiments that classify WHAT TYPES of mistakes the AI makes.

Experiment 1 (RAGTruth):  Auto-annotate each response for hallucination type
Experiment 2 (Huang):     Map each hallucination to standard taxonomy + find interview-specific types
Experiment 3 (DiaHaLu):   Check for dialogue-level issues (cross-turn, speaker confusion, etc.)

All 3 use GPT-4o-mini as the evaluator (judges any model's output).

Input:  {results_dir}/01_rag_responses.json
Output: {results_dir}/02_rq1_annotations.json
        {results_dir}/03_rq1_huang_taxonomy.json
        {results_dir}/04_rq1_diahalu_eval.json
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================
# EXPERIMENT 1: RAGTruth-style Annotation
# Paper: Niu et al., ACL 2024
# ============================================================

RAGTRUTH_PROMPT = """You are an expert annotator for hallucination detection in interview-based RAG systems.

TASK: Compare the AI's response against the original interview transcript. Find every mistake.

HALLUCINATION TYPES:
1. EVIDENT_CONFLICT — Response directly contradicts the transcript
2. SUBTLE_CONFLICT — Minor factual deviations from transcript
3. BASELESS_INFO — Claims not grounded in any part of the transcript
4. SENTIMENT_MISREPRESENTATION — Participant's tone or sentiment mischaracterized
5. REFUSAL_HALLUCINATION — Says "information not available" when it IS in the transcript
6. ROLE_ATTRIBUTION_DRIFT — Response starts with correct speaker attribution but progressively
   drifts to wrong speaker across consecutive sentences. Mark when 3+ sentences in the latter
   half attribute to wrong speaker while sentence 1 was correct. DISTINCT from single-sentence
   misattribution — this is a sequential positional pattern caused by context position decay.

ORIGINAL TRANSCRIPT (ground truth):
{transcript}

CONTEXT CHUNKS (what the AI was given):
{context}

QUERY: {query}

AI RESPONSE (evaluate this):
{response}

INSTRUCTIONS:
- Compare the response against the transcript word by word
- For each mistake, identify the exact text and its type
- Rate severity: LOW / MEDIUM / HIGH
- If the response says "not available" but the transcript has the answer, mark REFUSAL_HALLUCINATION

Return JSON:
{{
  "overall_label": "HALLUCINATED" or "FAITHFUL",
  "faithfulness_score": 0.0 to 1.0,
  "hallucinations": [
    {{
      "type": "one of the 6 types",
      "span": "exact text from response that is wrong",
      "evidence": "what the transcript actually says",
      "severity": "LOW/MEDIUM/HIGH",
      "explanation": "why this is a hallucination"
    }}
  ]
}}"""


def run_experiment_1(responses: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 1: RAGTruth-style annotation.
    For each RAG response, auto-annotate hallucination type, span, severity.
    Evaluator: GPT-4o-mini (regardless of which model generated the responses).
    """
    print("=" * 60)
    print("EXPERIMENT 1: RAGTruth Annotation")
    print("Paper: Niu et al., ACL 2024")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "02_rq1_annotations.json"

    client = openai.OpenAI()
    results = []
    valid = [r for r in responses if r.get("rag_response")]

    # Resume support
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}
        print(f"Resuming: {len(results)} already annotated")

    for i, r in enumerate(valid):
        key = (r["interview_id"], r["query_type"])
        if key in done_keys:
            continue

        print(f"\n[{i+1}/{len(valid)}] Annotating {r['query_type']}...")

        prompt = RAGTRUTH_PROMPT.format(
            transcript=r["transcript_text"][:4000],
            context=r["context"][:3000],
            query=r["query"],
            response=r["rag_response"],
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            annotation = json.loads(resp.choices[0].message.content)

            result = {
                "interview_id": r["interview_id"],
                "language": r["language"],
                "query": r["query"],
                "query_type": r["query_type"],
                "rag_model": r.get("rag_model", "unknown"),
                "rag_response": r["rag_response"],
                "overall_label": annotation.get("overall_label", "UNKNOWN"),
                "faithfulness_score": annotation.get("faithfulness_score", -1),
                "hallucinations": annotation.get("hallucinations", []),
                "num_hallucinations": len(annotation.get("hallucinations", [])),
                "hallucination_types": [h["type"] for h in annotation.get("hallucinations", [])],
            }
            results.append(result)
            done_keys.add(key)

            label = result["overall_label"]
            n = result["num_hallucinations"]
            types = result["hallucination_types"]
            print(f"  {label} | {n} issues | Types: {types}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "interview_id": r["interview_id"],
                "query_type": r["query_type"],
                "rag_model": r.get("rag_model", "unknown"),
                "error": str(e),
            })

        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} annotations to {output_path}")
    return results


# ============================================================
# EXPERIMENT 2: Huang Taxonomy Classification
# Paper: Huang et al., ACM TOIS 2024
# ============================================================

HUANG_PROMPT = """You are classifying hallucinations using the Huang et al. (2024) taxonomy.

HUANG TAXONOMY:
Level 1: FACTUALITY (contradicts world facts) or FAITHFULNESS (contradicts input context)
Level 2 under FACTUALITY: Factual_Fabrication, Factual_Inconsistency
Level 2 under FAITHFULNESS: Instruction_Inconsistency, Context_Inconsistency, Logical_Inconsistency

INTERVIEW-SPECIFIC EXTENSIONS (not in Huang):
- Speaker_Attribution_Error: correct content, wrong speaker
- Temporal_Flow_Error: events in wrong order
- Sentiment_Inference_Error: participant's tone mischaracterized
- Refusal_Error: refuses to answer when information IS available

For each hallucination below, classify it into Huang's taxonomy AND check if it's interview-specific.

HALLUCINATIONS TO CLASSIFY:
{hallucinations_json}

Return JSON:
{{
  "classifications": [
    {{
      "original_type": "the type from RAGTruth annotation",
      "huang_level1": "FACTUALITY or FAITHFULNESS",
      "huang_level2": "specific Huang category",
      "is_interview_specific": true/false,
      "interview_specific_type": "type name or null",
      "explanation": "why this classification"
    }}
  ],
  "summary": {{
    "total": number,
    "factuality_count": number,
    "faithfulness_count": number,
    "interview_specific_count": number,
    "interview_specific_percentage": number
  }}
}}"""


def run_experiment_2(annotations: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 2: Huang Taxonomy Classification.
    Maps every hallucination to standard taxonomy + identifies interview-specific types.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Huang Taxonomy Classification")
    print("Paper: Huang et al., ACM TOIS 2024")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    output_path = out_dir / "03_rq1_huang_taxonomy.json"

    client = openai.OpenAI()
    hallucinated = [a for a in annotations if a.get("overall_label") == "HALLUCINATED" and a.get("hallucinations")]

    if not hallucinated:
        print("No hallucinated responses to classify.")
        return []

    all_halls = []
    for a in hallucinated:
        for h in a["hallucinations"]:
            all_halls.append({
                "interview_id": a["interview_id"],
                "query_type": a["query_type"],
                "language": a["language"],
                "rag_model": a.get("rag_model", "unknown"),
                **h,
            })

    print(f"Classifying {len(all_halls)} hallucinations from {len(hallucinated)} responses...")

    results = []
    for batch_start in range(0, len(all_halls), 10):
        batch = all_halls[batch_start:batch_start+10]
        print(f"\n  Batch {batch_start//10 + 1}: classifying {len(batch)} hallucinations...")

        prompt = HUANG_PROMPT.format(
            hallucinations_json=json.dumps(batch, indent=2)[:4000]
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            results.append(result)

            s = result.get("summary", {})
            print(f"  Factuality: {s.get('factuality_count', '?')} | "
                  f"Faithfulness: {s.get('faithfulness_count', '?')} | "
                  f"Interview-specific: {s.get('interview_specific_count', '?')}")

        except Exception as e:
            print(f"  ERROR: {e}")

    with open(output_path, "w") as f:
        json.dump({"all_hallucinations": all_halls, "classifications": results}, f, indent=2)
    print(f"\nSaved taxonomy results to {output_path}")
    return results


# ============================================================
# EXPERIMENT 3: DiaHaLu Dialogue Evaluation
# Paper: Chen et al., EMNLP 2024
# ============================================================

DIAHALU_PROMPT = """You are evaluating a RAG response for DIALOGUE-SPECIFIC hallucination issues.

These are issues that only appear in conversational data (not in single documents):

1. CROSS_TURN_CONTRADICTION — Response contradicts something from an earlier part of the conversation
2. ENTITY_CONFUSION — Confuses two different entities or people mentioned in the interview
3. TOPIC_DRIFT — Response addresses a topic not discussed in the interview
4. TEMPORAL_DISTORTION — Gets the order of events/topics wrong
5. CONTEXT_FABRICATION — Fabricates interview context that doesn't exist
6. SPEAKER_CONFUSION — Attributes a statement to the wrong speaker (Agent vs Participant)

ORIGINAL INTERVIEW TRANSCRIPT:
{transcript}

QUERY: {query}

AI RESPONSE:
{response}

Evaluate the response for ALL 6 dialogue-specific issues.

Return JSON:
{{
  "is_faithful": true/false,
  "faithfulness_score": 0.0 to 1.0,
  "issues_found": [
    {{
      "type": "one of the 6 types",
      "description": "what went wrong",
      "severity": "LOW/MEDIUM/HIGH"
    }}
  ],
  "dialogue_quality_notes": "brief note on overall quality"
}}"""


def run_experiment_3(responses: List[Dict], results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Experiment 3: DiaHaLu Dialogue-Level Evaluation.
    Checks for conversation-specific hallucination patterns.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: DiaHaLu Dialogue Evaluation")
    print("Paper: Chen et al., EMNLP 2024")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    out_dir = Path(results_dir)
    output_path = out_dir / "04_rq1_diahalu_eval.json"

    client = openai.OpenAI()
    results = []
    valid = [r for r in responses if r.get("rag_response")]

    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
            results = existing.get("evaluations", [])
        done_keys = {(r["interview_id"], r["query_type"]) for r in results}
        print(f"Resuming: {len(results)} already evaluated")

    for i, r in enumerate(valid):
        key = (r["interview_id"], r["query_type"])
        if key in done_keys:
            continue

        print(f"\n[{i+1}/{len(valid)}] Evaluating dialogue issues...")

        prompt = DIAHALU_PROMPT.format(
            transcript=r["transcript_text"][:4000],
            query=r["query"],
            response=r["rag_response"],
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            evaluation = json.loads(resp.choices[0].message.content)

            result = {
                "interview_id": r["interview_id"],
                "language": r["language"],
                "query": r["query"],
                "query_type": r["query_type"],
                "rag_model": r.get("rag_model", "unknown"),
                "rag_response": r["rag_response"],
                "is_faithful": evaluation.get("is_faithful", None),
                "faithfulness_score": evaluation.get("faithfulness_score", -1),
                "issues_found": evaluation.get("issues_found", []),
                "num_issues": len(evaluation.get("issues_found", [])),
                "issue_types": [iss["type"] for iss in evaluation.get("issues_found", [])],
            }
            results.append(result)
            done_keys.add(key)

            status = "FAITHFUL" if result["is_faithful"] else f"ISSUES: {result['issue_types']}"
            print(f"  {status}")

        except Exception as e:
            print(f"  ERROR: {e}")

    summary = {
        "total_evaluated": len(results),
        "faithful": sum(1 for r in results if r.get("is_faithful")),
        "hallucinated": sum(1 for r in results if not r.get("is_faithful")),
        "by_language": {},
        "issue_type_counts": {},
    }

    for r in results:
        lang = r["language"]
        if lang not in summary["by_language"]:
            summary["by_language"][lang] = {"total": 0, "faithful": 0, "hallucinated": 0}
        summary["by_language"][lang]["total"] += 1
        if r.get("is_faithful"):
            summary["by_language"][lang]["faithful"] += 1
        else:
            summary["by_language"][lang]["hallucinated"] += 1

        for it in r.get("issue_types", []):
            summary["issue_type_counts"][it] = summary["issue_type_counts"].get(it, 0) + 1

    with open(output_path, "w") as f:
        json.dump({"evaluations": results, "summary": summary}, f, indent=2)
    print(f"\nSaved {len(results)} evaluations to {output_path}")
    print(f"Summary: {summary['faithful']} faithful, {summary['hallucinated']} hallucinated")
    return results


# ============================================================
# Run all RQ1 experiments
# ============================================================

def run_all_rq1(results_dir: str = "results/gpt-4o-mini"):
    """Run all 3 RQ1 experiments in sequence for the given model's results dir."""
    input_path = Path(results_dir) / "01_rag_responses.json"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate step first.")
        return

    with open(input_path) as f:
        responses = json.load(f)

    model_name = responses[0].get("rag_model", "unknown") if responses else "unknown"
    print(f"\nLoaded {len(responses)} RAG responses (model: {model_name})")

    annotations = run_experiment_1(responses, results_dir=results_dir)
    run_experiment_2(annotations, results_dir=results_dir)
    run_experiment_3(responses, results_dir=results_dir)

    print("\n" + "=" * 60)
    print(f"ALL RQ1 EXPERIMENTS COMPLETE — {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/gpt-4o-mini")
    args = parser.parse_args()
    run_all_rq1(results_dir=args.results_dir)
