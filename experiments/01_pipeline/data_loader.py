"""
Step 0: Data Loader
===================
Loads Convo interview data from Supabase CSV exports.
Parses JSON transcripts into clean conversation format.
Generates 6 types of research queries per interview.
Supports merging multiple CSV sources (real + synthetic).

Input:  One or more Supabase-format CSVs with transcript JSON
Output: List of interview dicts + list of query-context pairs
"""

import csv
import json
import sys
import random
from pathlib import Path
from typing import List, Dict

csv.field_size_limit(sys.maxsize)

# Default path — override via --data-sources in run_all.py
DEFAULT_CSV = Path(__file__).parent.parent / "data" / "real" / "supabase_responses.csv"


def load_interviews(csv_path: str = None, min_utterances: int = 10) -> List[Dict]:
    """
    Load and parse all usable interviews from a single CSV file.

    Returns list of dicts, each with:
        - id, study_id, language, is_complete
        - utterances: list of {speaker, text, start, end, confidence}
        - transcript_text: full conversation as readable string
        - chunks: list of speaker-aware text chunks for RAG context
        - source: which CSV file this came from
    """
    csv_path = csv_path or str(DEFAULT_CSV)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    source_name = Path(csv_path).name
    interviews = []
    for row in rows:
        raw = row.get("transcript", "")
        if not raw or raw.strip() in ("", "null"):
            continue

        try:
            data = json.loads(raw)
            utts_raw = data["body"]["payload"]["transcription"]["utterances"]
        except (json.JSONDecodeError, KeyError):
            continue

        if len(utts_raw) < min_utterances:
            continue

        utterances = []
        for u in utts_raw:
            utterances.append({
                "speaker": "Agent" if u.get("speaker") == 0 else "Participant",
                "text": u.get("text", "").strip(),
                "start": u.get("start", 0),
                "end": u.get("end", 0),
                "confidence": u.get("confidence", 0),
            })

        transcript_lines = [f'{u["speaker"]}: {u["text"]}' for u in utterances]
        transcript_text = "\n".join(transcript_lines)
        chunks = build_speaker_aware_chunks(utterances)

        interviews.append({
            "id": row.get("id", ""),
            "study_id": row.get("study_id", ""),
            "language": row.get("language", "en"),
            "is_complete": row.get("is_complete", "false") == "true",
            "n_utterances": len(utterances),
            "utterances": utterances,
            "transcript_text": transcript_text,
            "chunks": chunks,
            "source": source_name,
        })

    print(f"  [{source_name}] {len(interviews)} interviews loaded "
          f"({sum(1 for i in interviews if i['language']=='en')} EN, "
          f"{sum(1 for i in interviews if i['language']=='nl')} NL)")
    return interviews


def merge_csv_sources(csv_paths: List[str], min_utterances: int = 10) -> List[Dict]:
    """
    Load and merge interviews from multiple CSV files into one list.
    Deduplicates by interview id.

    csv_paths: list of paths to Supabase-format CSV files
    """
    print(f"\nLoading data from {len(csv_paths)} source(s)...")
    all_interviews = []
    seen_ids = set()
    for path in csv_paths:
        interviews = load_interviews(csv_path=path, min_utterances=min_utterances)
        for iv in interviews:
            if iv["id"] not in seen_ids:
                seen_ids.add(iv["id"])
                all_interviews.append(iv)
            else:
                print(f"  Skipping duplicate id: {iv['id'][:12]}...")

    en = sum(1 for i in all_interviews if i["language"] == "en")
    nl = sum(1 for i in all_interviews if i["language"] == "nl")
    print(f"  Merged total: {len(all_interviews)} unique interviews ({en} EN, {nl} NL)\n")
    return all_interviews


def build_speaker_aware_chunks(utterances: List[Dict], max_chunk_chars: int = 1500) -> List[str]:
    """
    Split utterances into chunks that respect speaker turns.
    Each chunk keeps full speaker turns together.
    """
    chunks = []
    current_chunk = []
    current_len = 0

    for u in utterances:
        line = f'{u["speaker"]}: {u["text"]}'
        line_len = len(line)

        if current_len + line_len > max_chunk_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(line)
        current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def generate_queries(interview: Dict) -> List[Dict]:
    """
    Generate 6 research query types for one interview.
    These are the questions the RAG system will try to answer.
    """
    utts = interview["utterances"]
    participant_utts = [u for u in utts if u["speaker"] == "Participant"]

    queries = []

    # 1. Speaker attribution
    queries.append({
        "query_type": "speaker_attribution",
        "query": "What did the interviewer (Agent) say at the beginning of this conversation?",
    })

    # 2. Participant content
    queries.append({
        "query_type": "participant_content",
        "query": "What topics did the participant discuss during this interview?",
    })

    # 3. Factual summary
    queries.append({
        "query_type": "factual_summary",
        "query": "Summarize the main points discussed in this interview.",
    })

    # 4. Specific content (referencing a real quote)
    if participant_utts:
        ref = random.choice(participant_utts)
        preview = ref["text"][:60]
        queries.append({
            "query_type": "specific_content",
            "query": f"What was discussed after the participant said '{preview}'?",
        })
    else:
        queries.append({
            "query_type": "specific_content",
            "query": "What specific statements did the participant make?",
        })

    # 5. Sentiment
    queries.append({
        "query_type": "sentiment",
        "query": "What was the overall sentiment or tone of the participant during this interview?",
    })

    # 6. Temporal ordering
    queries.append({
        "query_type": "temporal",
        "query": "What was the last topic discussed before the interview ended?",
    })

    return queries


def prepare_all_samples(interviews: List[Dict] = None,
                        max_interviews: int = None) -> List[Dict]:
    """
    Generate all query-context pairs for experiments.

    Returns list of dicts with:
        - interview_id, language, source, query, query_type
        - context (chunks joined as RAG context)
        - transcript_text (full ground truth)
    """
    if interviews is None:
        interviews = load_interviews()

    if max_interviews:
        interviews = interviews[:max_interviews]

    samples = []
    for interview in interviews:
        context = "\n\n---\n\n".join(interview["chunks"])
        queries = generate_queries(interview)

        for q in queries:
            samples.append({
                "interview_id": interview["id"],
                "language": interview["language"],
                "source": interview.get("source", "unknown"),
                "query": q["query"],
                "query_type": q["query_type"],
                "context": context,
                "transcript_text": interview["transcript_text"],
            })

    print(f"Prepared {len(samples)} query-context pairs from {len(interviews)} interviews")
    return samples


if __name__ == "__main__":
    # Test merging all data sources
    sources = [
        "data/real/supabase_responses.csv",
        "data/synthetic/synthetic_interviews.csv",
        "data/synthetic/supbase_english_synthetic.csv",
        "data/synthetic/supbase_dutch_synthetic.csv",
    ]
    interviews = merge_csv_sources(sources)
    samples = prepare_all_samples(interviews, max_interviews=5)
    print(f"\nFirst sample query: {samples[0]['query']}")
    print(f"From source: {samples[0]['source']}")
