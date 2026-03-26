"""
sentiment_utils.py — Sentiment analysis utilities.

Adapted from convo-backend-v2 SentimentEngine (proprietary, 2024).

Two components:
1. SENTIMENT_SYSTEM_PROMPT  — the exact prompt used in the Convo production system.
   Use this as your baseline to compare against chain-of-thought alternatives.

2. analyze_sentiment_local(text) — keyword-based fallback, zero dependencies.
   Use when no API key is available, or as a fast pre-filter.

Usage
-----
    from convo_utils.sentiment_utils import analyze_sentiment_local, SENTIMENT_SYSTEM_PROMPT

    # Local (no API)
    result = analyze_sentiment_local("I hate waiting for the results.")
    # → {'label': 'negative', 'score': -1.0, 'confidence': 0.4}

    # With OpenAI (production prompt)
    import openai, os, json
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT.format(language="nl")},
            {"role": "user", "content": "Het was best oké maar niet geweldig."},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)

Thesis use
----------
- Compare SENTIMENT_SYSTEM_PROMPT (baseline) vs. chain-of-thought prompt on
  76% hallucination rate sentiment queries to measure improvement.
- Use analyze_sentiment_local() as a fast pre-screen: if a text has no
  sentiment words at all (confidence=0.3), flag it for closer inspection.
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Production system prompt (from Convo SentimentEngine)
# Use as baseline in thesis ablation config A5 comparison
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment analysis engine for qualitative interview responses.
Analyze the participant's utterance and return ONLY a JSON object with these fields:
- "label": one of "very_positive", "positive", "neutral", "negative", "very_negative"
- "score": a float from -1.0 (very negative) to 1.0 (very positive)
- "confidence": a float from 0.0 to 1.0 indicating your confidence

Consider:
- The utterance is from a research interview, so context matters
- Detect subtle sentiment: frustration, enthusiasm, disappointment, satisfaction
- If the utterance is purely factual with no emotional content, label it "neutral"
- Consider cultural context for the language: {language}

Return ONLY the JSON object, no other text."""

# Chain-of-thought variant for ablation config A5
# Forces the model to quote evidence before concluding — reduces hallucination
SENTIMENT_COT_PROMPT = """You are a sentiment analysis engine for qualitative interview responses.

IMPORTANT: Always find evidence in the text before stating the sentiment.

Step 1: Find the exact words or phrases that indicate sentiment.
Step 2: Based only on those words, decide the sentiment.
Step 3: Return a JSON object with these fields:
- "evidence": the exact quote(s) from the text that show the sentiment
- "label": one of "very_positive", "positive", "neutral", "negative", "very_negative"
- "score": a float from -1.0 (very negative) to 1.0 (very positive)
- "confidence": a float from 0.0 to 1.0

Rules:
- If you cannot find direct evidence, set label to "neutral" and confidence below 0.4
- Never infer sentiment beyond what the words actually say
- Consider the language: {language}

Return ONLY the JSON object, no other text."""

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "great", "love", "excellent", "amazing", "good", "happy", "enjoy",
    "fantastic", "wonderful", "easy", "helpful", "satisfied", "appreciate",
    "perfect", "awesome", "best", "nice", "glad", "pleased", "positive",
    "clear", "smooth", "efficient", "fast", "reliable", "useful",
}

_NEGATIVE_WORDS = {
    "bad", "hate", "terrible", "awful", "poor", "frustrating", "difficult",
    "annoying", "disappointed", "confusing", "worst", "angry", "useless",
    "slow", "broken", "wrong", "unclear", "hard", "complicated", "boring",
    "waste", "failed", "error", "problem", "issue", "bug", "crash",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_sentiment_local(text: str) -> Dict[str, object]:
    """
    Keyword-based sentiment analysis — no API calls, no dependencies.

    Parameters
    ----------
    text : str
        Text to analyze (participant utterance or AI answer).

    Returns
    -------
    dict with keys:
        label      : str  — one of 'very_positive', 'positive', 'neutral',
                            'negative', 'very_negative'
        score      : float — -1.0 (very negative) to +1.0 (very positive)
        confidence : float — 0.0 to 1.0 (keyword-based = max 0.4)
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "confidence": 0.0}

    text_lower = text.lower()
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
    total = pos_count + neg_count

    if total == 0:
        return {"label": "neutral", "score": 0.0, "confidence": 0.3}

    score = (pos_count - neg_count) / max(total, 1)
    score = max(-1.0, min(1.0, score))

    if score > 0.5:
        label = "very_positive"
    elif score > 0.1:
        label = "positive"
    elif score < -0.5:
        label = "very_negative"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": round(score, 3), "confidence": 0.4}


def score_to_label(score: float) -> str:
    """Convert a numeric score (-1 to 1) to a sentiment label string."""
    if score > 0.5:
        return "very_positive"
    elif score > 0.1:
        return "positive"
    elif score < -0.5:
        return "very_negative"
    elif score < -0.1:
        return "negative"
    return "neutral"
