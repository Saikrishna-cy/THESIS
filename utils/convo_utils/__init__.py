"""
convo_utils — Reusable utilities from the Convo production backend.
Adapted from convo-backend-v2 (proprietary, 2024) with permission.

Modules:
    depth_scoring    — Answer completeness/depth scoring (zero dependencies)
    sentiment_utils  — Sentiment analysis: local fallback + production prompt
    translator       — Text and question translation via GPT-4o-mini
    embeddings/      — Embedding provider factory (local / OpenAI / hybrid)
"""
