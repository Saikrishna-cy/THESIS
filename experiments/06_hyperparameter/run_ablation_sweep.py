"""
Ablation Sweep — 10 Configurations from rag_configs.py
================================================================
PURPOSE:
  Runs all 10 predefined configurations from rag_configs.py and measures
  how each combination of prompt + retrieval settings affects hallucination rate.

WHY THIS MATTERS:
  rag_configs.py already defines a 3-group ablation grid:
    Group A: 4 prompt variants (same retrieval — isolates prompt effect)
    Group B: 3 chunk/topk combos (same prompt — isolates retrieval effect)
    Group C: 3 combined best configs (finds optimal combination)
  Running all 10 lets you write in your thesis:
  "Prompt engineering alone reduced hallucination by Xpp, retrieval alone by Ypp,
   and the combined approach by Zpp (Table 6.X)."

RESEARCH QUESTIONS ANSWERED:
  - Which prompt variant reduces hallucination most?
  - Which chunk_size / top_k combination gives lowest hallucination?
  - Does combining prompt + retrieval improvements give additive benefit?
  - Which hallucination types are most sensitive to retrieval configuration?

HOW TO RUN:
  # Quick test (3 interviews, GPT-4o-mini only):
  python D:\\RAG_THESIS\\hyperparameter_sweep\\run_ablation_sweep.py --max-interviews 3

  # Full run (all interviews):
  python D:\\RAG_THESIS\\hyperparameter_sweep\\run_ablation_sweep.py

PREREQUISITES:
  pip install openai
  OPENAI_API_KEY in D:\\thesis_hallucination\\.env
  D:\\thesis_hallucination\\data\\ must contain interview CSVs

ESTIMATED COST:
  3 interviews × 6 queries × 10 configs × 2 API calls = 360 calls ≈ $1-3
  Full 25 interviews: ≈ $10-25

OUTPUT:
  D:\\RAG_THESIS\\output\\sweep_results.json
  D:\\RAG_THESIS\\output\\sweep_comparison_table.csv

HOW IT HELPS YOUR IEEE PAPER:
  This IS your ablation study (Section 6). The comparison table becomes
  Table 6.X in your thesis. You can directly compare baseline vs best config
  and claim: "Configuration C2 reduces hallucination by Xpp over baseline."
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
THESIS_SRC = Path(r"D:\thesis_hallucination\src")
THESIS_DATA = Path(r"D:\thesis_hallucination\data")
sys.path.insert(0, str(THESIS_SRC))

_ENV = Path(r"D:\thesis_hallucination\.env")
if _ENV.exists():
    for line in _ENV.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

OUTPUT_DIR = Path(r"D:\RAG_THESIS\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import openai
    client = openai.OpenAI()
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

from rag_configs import CONFIGS, PROMPT_TEMPLATES

# ── annotation prompt (E1 style) ─────────────────────────────────────────────
ANNOTATION_PROMPT = """\
You are an expert annotator for hallucination detection in interview-based RAG systems.

Compare the AI's response against the original interview transcript.

HALLUCINATION TYPES:
1. EVIDENT_CONFLICT — Directly contradicts the transcript
2. SUBTLE_CONFLICT — Minor factual deviations
3. BASELESS_INFO — Claims not in any part of the transcript
4. SPEAKER_MISATTRIBUTION — Wrong speaker credited
5. TEMPORAL_CONFUSION — Events in wrong order
6. SENTIMENT_MISREPRESENTATION — Tone/sentiment mischaracterized
7. REFUSAL_HALLUCINATION — Says "not available" when it IS in transcript

ORIGINAL TRANSCRIPT:
{transcript}

QUERY: {query}

AI RESPONSE:
{response}

Return JSON:
{{
  "overall_label": "HALLUCINATED" or "FAITHFUL",
  "faithfulness_score": 0.0 to 1.0,
  "hallucinations": [
    {{"type": "one of 7 types", "span": "exact text", "severity": "LOW/MEDIUM/HIGH"}}
  ]
}}"""


def embed_text(text: str) -> list:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return resp.data[0].embedding


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def chunk_utterances(utterances, chunk_size):
    chunks, cur, cur_len = [], [], 0
    for u in utterances:
        line = f'{u["speaker"]}: {u["text"]}'
        if cur_len + len(line) > chunk_size and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [""]


def retrieve_context(interview, config):
    """Build context string according to config's retrieval settings."""
    if config["retrieval"] == "full_context":
        return interview["transcript_text"][:4000]

    # semantic_topk: chunk + embed + retrieve top-k
    chunks = chunk_utterances(interview["utterances"], config["chunk_size"])
    if len(chunks) <= config["top_k"]:
        return "\n\n---\n\n".join(chunks)

    # Embed all chunks (simple cache: just do it)
    chunk_embs = []
    for c in chunks:
        chunk_embs.append(embed_text(c))
        time.sleep(0.05)

    # Use first query type embedding as representative
    q_emb = embed_text(f"What happened in this interview?")
    scored = sorted(zip([cosine_sim(q_emb, e) for e in chunk_embs], chunks), reverse=True)
    top = [c for _, c in scored[:config["top_k"]]]
    return "\n\n---\n\n".join(top)


def generate_response(context, query, prompt_variant, temperature=0.3):
    system = PROMPT_TEMPLATES[prompt_variant].format(context=context)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": query}],
        temperature=temperature,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def annotate(transcript, query, response):
    prompt = ANNOTATION_PROMPT.format(
        transcript=transcript[:3000], query=query, response=response
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=700,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def load_interviews(max_n=None):
    """Load interviews from thesis data directory."""
    try:
        from data_loader import merge_csv_sources
        sources = [
            str(THESIS_DATA / "real" / "supabase_responses.csv"),
            str(THESIS_DATA / "synthetic" / "synthetic_interviews.csv"),
            str(THESIS_DATA / "synthetic" / "supbase_english_synthetic.csv"),
            str(THESIS_DATA / "synthetic" / "supbase_dutch_synthetic.csv"),
        ]
        existing = [s for s in sources if Path(s).exists()]
        if not existing:
            print("ERROR: No data files found.")
            sys.exit(1)
        interviews = merge_csv_sources(existing)
        if max_n:
            interviews = interviews[:max_n]
        return interviews
    except Exception as e:
        print(f"ERROR loading interviews: {e}")
        sys.exit(1)


QUERIES = [
    ("speaker_attribution", "What did the interviewer (Agent) say at the beginning?"),
    ("sentiment", "What was the overall sentiment or tone of the participant?"),
    ("factual_summary", "Summarize the main points discussed in this interview."),
    ("temporal", "What was the last topic discussed before the interview ended?"),
]


def run_config(config, interviews):
    """Run one config across all interviews × query types."""
    results = []
    for interview in interviews:
        for qtype, query in QUERIES:
            try:
                context = retrieve_context(interview, config)
                time.sleep(0.3)
                response = generate_response(context, query, config["prompt_variant"])
                time.sleep(0.5)
                ann = annotate(interview["transcript_text"], query, response)
                time.sleep(0.5)
                results.append({
                    "config": config["name"],
                    "group": config["group"],
                    "interview_id": interview["id"],
                    "language": interview["language"],
                    "query_type": qtype,
                    "label": ann.get("overall_label", "UNKNOWN"),
                    "faithfulness": ann.get("faithfulness_score", 0),
                    "n_issues": len(ann.get("hallucinations", [])),
                    "issue_types": [h["type"] for h in ann.get("hallucinations", [])],
                })
            except Exception as e:
                print(f"  ERROR {config['name']} {qtype}: {e}")
                results.append({
                    "config": config["name"], "group": config["group"],
                    "interview_id": interview.get("id", ""), "query_type": qtype,
                    "label": "ERROR", "faithfulness": 0, "n_issues": 0, "issue_types": [],
                })
    return results


def compute_summary(results, config_name):
    valid = [r for r in results if r["label"] in ("HALLUCINATED", "FAITHFUL")]
    if not valid:
        return {}
    n = len(valid)
    hall_rate = sum(1 for r in valid if r["label"] == "HALLUCINATED") / n
    avg_faith = sum(r["faithfulness"] for r in valid) / n
    type_counts = {}
    for r in valid:
        for t in r["issue_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "config": config_name,
        "n": n,
        "hallucination_rate": round(hall_rate, 4),
        "avg_faithfulness": round(avg_faith, 4),
        "hallucination_type_counts": type_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-interviews", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION SWEEP — 10 CONFIGURATIONS FROM rag_configs.py")
    print("=" * 60)

    interviews = load_interviews(args.max_interviews)
    print(f"Using {len(interviews)} interviews × {len(QUERIES)} query types = "
          f"{len(interviews)*len(QUERIES)} evaluations per config")
    print(f"Total API calls: ~{len(interviews)*len(QUERIES)*len(CONFIGS)*2} (2 per eval)\n")

    all_results = []
    summaries = []

    for cfg in CONFIGS:
        print(f"\n{'='*50}")
        print(f"Running config: {cfg['name']} | {cfg['description']}")
        print(f"{'='*50}")
        results = run_config(cfg, interviews)
        all_results.extend(results)
        summary = compute_summary(results, cfg["name"])
        summaries.append({**summary, **cfg})

        n = summary.get("n", 0)
        hall = summary.get("hallucination_rate", 0)
        faith = summary.get("avg_faithfulness", 0)
        print(f"  → Hall. rate: {hall:.1%} | Avg faithfulness: {faith:.3f} | N={n}")

    # ── find best config ──────────────────────────────────────────────────────
    valid_summaries = [s for s in summaries if "hallucination_rate" in s]
    if valid_summaries:
        best = min(valid_summaries, key=lambda x: x["hallucination_rate"])
        baseline = next((s for s in valid_summaries if s["config"] == "A1_control"), None)

    # ── print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Config':<25} {'Group':<18} {'Hall%':>6} {'Faith.':>7} {'vs A1':>8}")
    print("-" * 80)
    base_rate = baseline["hallucination_rate"] if baseline else None
    for s in valid_summaries:
        delta = ""
        if base_rate is not None and s["config"] != "A1_control":
            diff = (s["hallucination_rate"] - base_rate) * 100
            delta = f"{diff:+.1f}pp"
        print(f"  {s['config']:<23} {s.get('group',''):<18} "
              f"{s['hallucination_rate']:>5.1%} {s['avg_faithfulness']:>7.3f} {delta:>8}")
    print("=" * 80)
    if valid_summaries:
        print(f"\nBest config: {best['config']} (Hall. rate = {best['hallucination_rate']:.1%})")
        if baseline:
            improvement = (baseline["hallucination_rate"] - best["hallucination_rate"]) * 100
            print(f"vs baseline (A1): {improvement:+.1f}pp improvement")

    # ── save results ──────────────────────────────────────────────────────────
    output = {
        "n_configs": len(CONFIGS),
        "n_interviews": len(interviews),
        "n_queries_per_config": len(interviews) * len(QUERIES),
        "summaries": valid_summaries,
        "best_config": best.get("config") if valid_summaries else None,
        "all_results": all_results,
    }
    with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── CSV table ─────────────────────────────────────────────────────────────
    import csv
    with open(OUTPUT_DIR / "sweep_comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "group", "description",
                                                "prompt_variant", "chunk_size", "top_k",
                                                "hallucination_rate", "avg_faithfulness"])
        writer.writeheader()
        for s in valid_summaries:
            writer.writerow({
                "config": s.get("config"), "group": s.get("group"),
                "description": s.get("description"), "prompt_variant": s.get("prompt_variant"),
                "chunk_size": s.get("chunk_size"), "top_k": s.get("top_k"),
                "hallucination_rate": s.get("hallucination_rate"),
                "avg_faithfulness": s.get("avg_faithfulness"),
            })

    print(f"\nResults saved:")
    print(f"  {OUTPUT_DIR / 'sweep_results.json'}")
    print(f"  {OUTPUT_DIR / 'sweep_comparison_table.csv'}")

    print("\nUSE IN THESIS (Section 6 — Ablation Study):")
    if baseline and valid_summaries:
        print(f'  "We evaluate 10 configurations across three intervention groups.')
        print(f"  The baseline prompt (A1) achieves {baseline['hallucination_rate']:.1%} hallucination.")
        print(f"  The best configuration ({best['config']}) reduces this to")
        print(f"  {best['hallucination_rate']:.1%} (−{improvement:.1f}pp), demonstrating that")
        print(f'  [prompt engineering / retrieval configuration / combined] is most effective."')


if __name__ == "__main__":
    main()
