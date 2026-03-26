# convo_utils

Reusable utilities extracted and adapted from the **Convo production backend**
(`convo-backend-v2`, proprietary, 2024) for use in the thesis project.

> **Credit**: Depth scoring and embedding infrastructure adapted from Convo (2024) with permission.
> Sentiment prompt reproduced from production system for baseline comparison.

---

## Modules

### `depth_scoring.py` — Zero dependencies
Measures how thorough an answer is (0 = empty/shallow, 1 = thorough).

```python
from convo_utils.depth_scoring import score_completeness, extract_keywords

score = score_completeness("I really like it because the workflow is clear and saves time.")
# → 0.712

keywords = extract_keywords("The onboarding process was confusing at first.")
# → ['onboarding', 'process', 'confusing']
```

**Thesis use**: Flag AI answers with `score < 0.3` as potential hallucination risk.
Low-depth answers correlate with fabricated or evasive responses.

---

### `sentiment_utils.py` — Zero dependencies for local mode
Two sentiment tools:
- `SENTIMENT_SYSTEM_PROMPT` — exact prompt from Convo production system (use as ablation baseline)
- `SENTIMENT_COT_PROMPT` — chain-of-thought variant (ablation config A5, forces evidence quoting)
- `analyze_sentiment_local(text)` — keyword-based, no API needed

```python
from convo_utils.sentiment_utils import analyze_sentiment_local, SENTIMENT_SYSTEM_PROMPT

result = analyze_sentiment_local("The product is frustrating and confusing.")
# → {'label': 'negative', 'score': -1.0, 'confidence': 0.4}
```

**Thesis use**: Compare `SENTIMENT_SYSTEM_PROMPT` (baseline 76% hallucination) vs
`SENTIMENT_COT_PROMPT` (ablation A5) to measure improvement.

---

### `translator.py` — Requires `OPENAI_API_KEY`
Translates text and question lists via GPT-4o-mini.

```python
import asyncio
from convo_utils.translator import QuestionTranslator

tr = QuestionTranslator()

# Translate a transcript chunk (for translate-first RAG pipeline)
en = asyncio.run(tr.translate_text(
    "Ik vind het moeilijk om samen te werken.",
    from_lang="nl", to_lang="en"
))
# → "I find it difficult to collaborate."
```

**Thesis use**: Implement ablation Group D — translate-first RAG pipeline:
1. Translate Dutch transcript → English
2. Query model in English
3. Translate answer → Dutch
Expected result: reduces Dutch hallucination rate from 76.7% toward English baseline.

---

### `embeddings/` — Requires `sentence-transformers` or `openai`
Embedding provider factory supporting local, OpenAI, and hybrid modes.

```python
import asyncio
from convo_utils.embeddings.factory import EmbeddingProviderFactory
from convo_utils.embeddings.config import EmbeddingConfig

config = EmbeddingConfig(provider_type="local")
provider = asyncio.run(EmbeddingProviderFactory.create_and_initialize(config))
embedding = asyncio.run(provider.embed_text("How did you feel about the interview?"))
```

**Thesis use**: Use as the RAG retrieval embedding layer — already supports multilingual
`paraphrase-multilingual-MiniLM-L12-v2` model, which handles Dutch natively.

---

## Installation requirements

```
# Already in thesis venv:
openai          # for translator + sentiment OpenAI mode
sentence-transformers  # for embeddings local mode
numpy           # for embeddings
```

depth_scoring.py and the local parts of sentiment_utils.py have **zero dependencies**.
