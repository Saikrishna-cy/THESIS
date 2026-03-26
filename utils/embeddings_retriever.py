"""
Semantic Retrieval using OpenAI Embeddings
==========================================
Chunks transcripts at a given chunk_size with optional overlap, embeds with
text-embedding-3-small, and retrieves the top-k most relevant chunks using
cosine similarity (top_k) or MMR (Maximal Marginal Relevance).

New parameters vs original:
  chunk_overlap       — number of characters of overlap between consecutive chunks
                        (prevents information loss at chunk boundaries)
  retrieval_strategy  — "top_k" (default) or "mmr" (diversified retrieval)
  similarity_threshold — minimum cosine similarity to include a chunk (filters noise)

No external vector DB needed — pure Python.
"""

import math
import time
from typing import List, Dict, Optional

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

EMBED_MODEL = "text-embedding-3-small"
_embed_cache: dict = {}  # (interview_id, chunk_size, chunk_overlap) -> {"chunks": [...], "embeddings": [...]}


def chunk_utterances(
    utterances: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    Split utterances into chunks of at most chunk_size characters.
    Keeps full speaker turns together. When chunk_overlap > 0, consecutive
    chunks share the last N characters of the previous chunk, preventing
    information loss at boundaries (reduces CONTEXT_FABRICATION hallucinations).
    """
    # First pass: build non-overlapping base chunks
    base_chunks = []
    current_lines: List[str] = []
    current_len = 0

    for u in utterances:
        line = f'{u["speaker"]}: {u["text"]}'
        if current_len + len(line) > chunk_size and current_lines:
            base_chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line)

    if current_lines:
        base_chunks.append("\n".join(current_lines))

    if not base_chunks:
        return [""]

    if chunk_overlap <= 0 or len(base_chunks) == 1:
        return base_chunks

    # Second pass: add overlap by prepending tail of previous chunk
    overlapped = [base_chunks[0]]
    for i in range(1, len(base_chunks)):
        tail = base_chunks[i - 1][-chunk_overlap:] if len(base_chunks[i - 1]) > chunk_overlap else base_chunks[i - 1]
        overlapped.append(tail + "\n" + base_chunks[i])

    return overlapped


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_texts(texts: List[str], client) -> List[List[float]]:
    """
    Embed a list of texts using OpenAI text-embedding-3-small.
    Batches in groups of 100 to stay within API limits.
    """
    embeddings = []
    cleaned = [t.replace("\n", " ").strip() or " " for t in texts]

    for i in range(0, len(cleaned), 100):
        batch = cleaned[i : i + 100]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for item in response.data:
            embeddings.append(item.embedding)
        if i + 100 < len(cleaned):
            time.sleep(0.05)

    return embeddings


def get_interview_index(
    interview: Dict,
    chunk_size: int,
    client,
    chunk_overlap: int = 0,
) -> Dict:
    """
    Build (or retrieve from cache) chunk embeddings for one interview.
    Cache key: (interview_id, chunk_size, chunk_overlap).
    """
    cache_key = (interview["id"], chunk_size, chunk_overlap)
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    chunks = chunk_utterances(
        interview["utterances"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    embeddings = embed_texts(chunks, client)
    index = {"chunks": chunks, "embeddings": embeddings}
    _embed_cache[cache_key] = index
    return index


def _retrieve_mmr(
    query_emb: List[float],
    chunks: List[str],
    embeddings: List[List[float]],
    top_k: int,
    lambda_param: float = 0.5,
    similarity_threshold: float = 0.0,
) -> List[str]:
    """
    Maximal Marginal Relevance retrieval.
    Balances relevance to query (lambda_param) vs diversity among selected chunks
    (1 - lambda_param). Higher lambda = more like plain top_k. Lower = more diverse.

    lambda_param=0.5 is the standard MMR setting from Carbonell & Goldstein (1998).
    """
    scores = [_cosine_sim(query_emb, emb) for emb in embeddings]
    candidates = [
        (i, scores[i]) for i in range(len(chunks))
        if scores[i] >= similarity_threshold
    ]

    if not candidates:
        candidates = [(i, scores[i]) for i in range(len(chunks))]

    selected_indices = []
    while len(selected_indices) < top_k and candidates:
        if not selected_indices:
            best = max(candidates, key=lambda x: x[1])
            selected_indices.append(best[0])
            candidates = [c for c in candidates if c[0] != best[0]]
        else:
            best_mmr = None
            best_mmr_score = float("-inf")
            for idx, rel_score in candidates:
                max_sim_to_selected = max(
                    _cosine_sim(embeddings[idx], embeddings[s])
                    for s in selected_indices
                )
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim_to_selected
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_mmr = idx
            selected_indices.append(best_mmr)
            candidates = [c for c in candidates if c[0] != best_mmr]

    return [chunks[i] for i in selected_indices]


def retrieve_chunks(
    query: str,
    interview: Dict,
    chunk_size: int,
    top_k: int,
    client,
    chunk_overlap: int = 0,
    retrieval_strategy: str = "top_k",
    similarity_threshold: float = 0.0,
) -> List[str]:
    """
    Return the top-k most relevant chunks using the specified strategy.

    retrieval_strategy:
      "top_k" — standard cosine similarity ranking (default)
      "mmr"   — Maximal Marginal Relevance (diversified; reduces redundant chunks)

    similarity_threshold: minimum cosine similarity to include (0.0 = no filter).
    chunk_overlap: characters of overlap between consecutive chunks.
    """
    index = get_interview_index(interview, chunk_size, client, chunk_overlap=chunk_overlap)
    chunks = index["chunks"]
    embeddings = index["embeddings"]

    if not embeddings or len(chunks) <= top_k:
        return chunks

    query_emb = embed_texts([query], client)[0]

    if retrieval_strategy == "mmr":
        return _retrieve_mmr(
            query_emb, chunks, embeddings, top_k,
            similarity_threshold=similarity_threshold,
        )

    # Default: top_k by cosine similarity
    scored = [
        (_cosine_sim(query_emb, emb), chunk)
        for emb, chunk in zip(embeddings, chunks)
        if _cosine_sim(query_emb, emb) >= similarity_threshold
    ]
    if not scored:
        scored = [(_cosine_sim(query_emb, emb), chunk)
                  for emb, chunk in zip(embeddings, chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# Backwards-compatible alias
def retrieve_top_k(
    query: str, interview: Dict, chunk_size: int, top_k: int, client
) -> List[str]:
    """Legacy function — use retrieve_chunks() for new code."""
    return retrieve_chunks(query, interview, chunk_size, top_k, client)
