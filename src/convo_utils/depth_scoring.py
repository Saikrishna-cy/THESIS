"""
depth_scoring.py — Answer completeness / depth scoring utilities.

Adapted from convo-backend-v2 InterviewMemoryEngine (proprietary, 2024).
Zero external dependencies — pure Python.

Usage
-----
    from convo_utils.depth_scoring import score_completeness, extract_keywords

    score = score_completeness("I really enjoyed it because the onboarding was clear.")
    # → 0.623  (0 = shallow/empty, 1 = thorough/deep)

    keywords = extract_keywords("The product helped me save time on repetitive tasks.")
    # → ['product', 'helped', 'save', 'time', 'repetitive', 'tasks']

Thesis use
----------
- Flag answers with score < 0.3 as potential hallucination risk
  (low-depth answers → model is more likely to fabricate)
- Use as a 5th detection signal alongside SelfCheckGPT / MiniCheck / RAGAS / AlignScore
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they",
    "them", "their", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "doing", "would", "should", "could",
    "might", "shall", "will", "can", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "into", "through", "during", "before", "after", "to", "from",
    "up", "in", "out", "on", "over", "again", "then", "when", "where", "how",
    "all", "each", "few", "more", "most", "other", "some", "no", "not", "only",
    "so", "than", "too", "very", "just", "now", "also", "really", "like",
    "know", "think", "yeah", "yes", "um", "uh", "okay", "well", "got", "thing",
    "things", "lot", "bit", "say", "said", "kind", "sort", "right", "actually",
}

# Phrases that indicate a reasoned, specific, or deep answer
_DEPTH_SIGNALS: List[str] = [
    "because", "the reason", "specifically", "in particular", "especially",
    "for example", "for instance", "such as", "notably", "primarily",
    "due to", "as a result", "therefore", "however", "although",
    "compared to", "rather than", "on the other hand", "what i mean is",
    "the main", "the biggest", "the most important",
]

# Phrases that indicate a shallow or deflecting answer
_SHALLOW_SIGNALS: List[str] = [
    "i don't know", "not sure", "no idea", "nothing really", "i guess",
    "maybe", "probably", "not really", "not much", "not applicable",
    "i don't use it", "no comment", "pass", "i can't say",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_completeness(text: str) -> float:
    """
    Score answer depth on 0.0 (silent/shallow) to 1.0 (thorough) scale.

    Weights
    -------
    45%  word count        — more words = more content (ceiling at 70 words)
    25%  lexical richness  — unique meaningful words relative to total
    30%  depth signals     — presence of reasoning / example phrases
    -N   shallow penalty   — deflecting phrases reduce the score

    Parameters
    ----------
    text : str
        The AI-generated answer or participant response to score.

    Returns
    -------
    float
        0.0 = empty/completely shallow, 1.0 = thorough and detailed.
        A score below 0.3 suggests the answer may be fabricated or evasive.
    """
    if not text or not text.strip():
        return 0.0

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    # 1. Word-count component (linear, saturates at 70 words)
    wc_score = min(1.0, word_count / 70)

    # 2. Lexical richness
    meaningful = [w for w in words if len(w) > 3 and w not in _STOP_WORDS]
    unique_ratio = len(set(meaningful)) / max(len(meaningful), 1)
    richness_score = min(1.0, unique_ratio * 1.4)

    # 3. Depth signals
    depth_hits = sum(1 for phrase in _DEPTH_SIGNALS if phrase in text_lower)
    depth_score = min(1.0, depth_hits / 3)

    # 4. Shallow penalty
    shallow_hits = sum(1 for phrase in _SHALLOW_SIGNALS if phrase in text_lower)
    penalty = min(0.5, shallow_hits * 0.18)

    raw = wc_score * 0.45 + richness_score * 0.25 + depth_score * 0.30
    return round(max(0.0, min(1.0, raw - penalty)), 3)


def extract_keywords(text: str, top_n: int = 6) -> List[str]:
    """
    Return the top N non-stopword keywords from text by frequency.

    Parameters
    ----------
    text : str
        Input text.
    top_n : int
        Maximum number of keywords to return (default 6).

    Returns
    -------
    List[str]
        Sorted by frequency, most common first.
    """
    tokens = [t for t in _tokenize(text) if t not in _STOP_WORDS]
    if not tokens:
        return []
    return [word for word, _ in Counter(tokens).most_common(top_n)]


def summarize_extractive(utterances: List[str], max_chars: int = 220) -> str:
    """
    Extractive summarization without any external library.
    Scores sentences by shared keyword density and returns the top 2–3.

    Parameters
    ----------
    utterances : List[str]
        List of text segments (e.g. sentences or utterances) to summarize.
    max_chars : int
        Approximate character limit for the returned summary.

    Returns
    -------
    str
        A concise extractive summary.
    """
    full_text = " ".join(utterances)
    if len(full_text) <= max_chars:
        return full_text.strip()

    keywords = set(extract_keywords(full_text, 12))
    sentences = [s.strip() for s in re.split(r"[.!?]+", full_text) if len(s.strip()) >= 8]

    scored = []
    for sent in sentences:
        tokens = set(_tokenize(sent))
        score = len(tokens & keywords)
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    result, chars = [], 0
    for _, sent in scored:
        if chars + len(sent) > max_chars:
            break
        result.append(sent)
        chars += len(sent)

    return ". ".join(result[:3]).strip() or full_text[:max_chars]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())
