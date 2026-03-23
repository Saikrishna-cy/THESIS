"""
Semantic Retrieval using OpenAI Embeddings
==========================================
Chunks transcripts at a given chunk_size, embeds with text-embedding-3-small,
and retrieves the top-k most relevant chunks for a query using cosine similarity.
No external vector DB needed — pure Python + numpy.
"""

import math
import time
from typing import List, Dict

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

EMBED_MODEL = "text-embedding-3-small"
_embed_cache: dict = {}  # (interview_id, chunk_size) -> {"chunks": [...], "embeddings": [...]}


def chunk_utterances(utterances: List[Dict], chunk_size: int = 1000) -> List[str]:
    """
    Split utterances into chunks of at most chunk_size characters.
    Keeps full speaker turns together — never splits mid-utterance.
    """
    chunks = []
    current_lines: List[str] = []
    current_len = 0

    for u in utterances:
        line = f'{u["speaker"]}: {u["text"]}'
        if current_len + len(line) > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line)

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks if chunks else [""]


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


def get_interview_index(interview: Dict, chunk_size: int, client) -> Dict:
    """
    Build (or retrieve from cache) chunk embeddings for one interview.
    Cache key: (interview_id, chunk_size).
    """
    cache_key = (interview["id"], chunk_size)
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    chunks = chunk_utterances(interview["utterances"], chunk_size=chunk_size)
    embeddings = embed_texts(chunks, client)
    index = {"chunks": chunks, "embeddings": embeddings}
    _embed_cache[cache_key] = index
    return index


def retrieve_top_k(
    query: str,
    interview: Dict,
    chunk_size: int,
    top_k: int,
    client,
) -> List[str]:
    """
    Return the top-k most relevant chunks for the query using cosine similarity.
    """
    index = get_interview_index(interview, chunk_size, client)
    chunks = index["chunks"]
    embeddings = index["embeddings"]

    if not embeddings or len(chunks) <= top_k:
        return chunks

    query_emb = embed_texts([query], client)[0]
    scored = [
        (_cosine_sim(query_emb, emb), chunk)
        for emb, chunk in zip(embeddings, chunks)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
