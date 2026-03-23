"""
Parameter Grid for Thesis: Reducing Hallucinations in Interview-Based RAG
=========================================================================
10 configurations across 3 intervention groups — maps to a clean ablation
table in the thesis:

  Group A (4): Prompt engineering only — control + 3 prompt variants
  Group B (3): Retrieval improvement only — vary chunk size, baseline prompt
  Group C (3): Combined — semantic retrieval + best prompt variants

This structure lets you write: "Prompt engineering alone reduced hallucinations
by X%, retrieval alone by Y%, and the combined approach by Z%."
"""

# ── Prompt Templates ──────────────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "baseline": (
        "You are a research assistant analyzing qualitative interview data. "
        "Answer the user's question based ONLY on the interview transcript provided below. "
        "If the answer is not in the transcript, say 'This information is not available in the transcript.' "
        "Be specific and cite the speaker (Agent or Participant) when referencing statements.\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),
    "anti_refusal": (
        "You are a research assistant analyzing qualitative interview data. "
        "Answer the user's question based ONLY on the interview transcript below.\n"
        "CRITICAL RULE: If ANY relevant content exists in the transcript, you MUST use it. "
        "Do NOT claim information is unavailable unless the transcript is completely silent on the topic. "
        "A partial answer grounded in the transcript is always better than a refusal. "
        "Cite the speaker (Agent or Participant) for every statement you reference.\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),
    "citation_strict": (
        "You are a research assistant analyzing qualitative interview data.\n"
        "STRICT RULES:\n"
        "1. Answer ONLY from the transcript below — never add outside knowledge.\n"
        "2. Every factual claim MUST be attributed: write 'Agent said: ...' or 'Participant said: ...'\n"
        "3. Do NOT paraphrase speaker intent — quote or closely paraphrase actual words.\n"
        "4. If information is truly absent from the transcript, say so once, briefly.\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),
    "conservative": (
        "You are a careful research assistant analyzing interview transcripts.\n"
        "Answer ONLY from the transcript below. Apply these rules:\n"
        "- State only what is clearly and directly supported by the text.\n"
        "- Use 'The transcript shows...' or 'According to [Speaker]:' to signal grounding.\n"
        "- For uncertain inferences write 'The transcript suggests...' not 'X felt/believed...'\n"
        "- Always name the speaker (Agent or Participant) when referencing a statement.\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),
}

# ── Configuration Grid ────────────────────────────────────────────────────────
CONFIGS = [
    # ── Group A: Prompt Engineering only (full transcript as context) ──────
    {
        "name": "A1_control",
        "group": "A_prompt_only",
        "description": "Control: current baseline setup (no changes)",
        "retrieval": "full_context",
        "chunk_size": 1500,
        "top_k": None,
        "prompt_variant": "baseline",
    },
    {
        "name": "A2_anti_refusal",
        "group": "A_prompt_only",
        "description": "Prompt engineering: anti-refusal instruction added",
        "retrieval": "full_context",
        "chunk_size": 1500,
        "top_k": None,
        "prompt_variant": "anti_refusal",
    },
    {
        "name": "A3_citation_strict",
        "group": "A_prompt_only",
        "description": "Prompt engineering: strict speaker citation enforced",
        "retrieval": "full_context",
        "chunk_size": 1500,
        "top_k": None,
        "prompt_variant": "citation_strict",
    },
    {
        "name": "A4_conservative",
        "group": "A_prompt_only",
        "description": "Prompt engineering: conservative hedging language",
        "retrieval": "full_context",
        "chunk_size": 1500,
        "top_k": None,
        "prompt_variant": "conservative",
    },

    # ── Group B: Retrieval Improvement only (baseline prompt) ──────────────
    {
        "name": "B1_rag_c500_k3",
        "group": "B_retrieval_only",
        "description": "Semantic RAG: small chunks (500 chars), top-3 retrieved",
        "retrieval": "semantic_topk",
        "chunk_size": 500,
        "top_k": 3,
        "prompt_variant": "baseline",
    },
    {
        "name": "B2_rag_c1000_k3",
        "group": "B_retrieval_only",
        "description": "Semantic RAG: medium chunks (1000 chars), top-3 retrieved",
        "retrieval": "semantic_topk",
        "chunk_size": 1000,
        "top_k": 3,
        "prompt_variant": "baseline",
    },
    {
        "name": "B3_rag_c1500_k3",
        "group": "B_retrieval_only",
        "description": "Semantic RAG: large chunks (1500 chars), top-3 retrieved",
        "retrieval": "semantic_topk",
        "chunk_size": 1500,
        "top_k": 3,
        "prompt_variant": "baseline",
    },

    # ── Group C: Combined — Retrieval + Prompt ─────────────────────────────
    {
        "name": "C1_rag_c1000_k3_anti_refusal",
        "group": "C_combined",
        "description": "Combined: medium chunks, top-3, anti-refusal prompt",
        "retrieval": "semantic_topk",
        "chunk_size": 1000,
        "top_k": 3,
        "prompt_variant": "anti_refusal",
    },
    {
        "name": "C2_rag_c1000_k3_citation_strict",
        "group": "C_combined",
        "description": "Combined: medium chunks, top-3, strict citation prompt",
        "retrieval": "semantic_topk",
        "chunk_size": 1000,
        "top_k": 3,
        "prompt_variant": "citation_strict",
    },
    {
        "name": "C3_rag_c500_k5_anti_refusal",
        "group": "C_combined",
        "description": "Combined: fine-grained chunks (500), top-5, anti-refusal prompt",
        "retrieval": "semantic_topk",
        "chunk_size": 500,
        "top_k": 5,
        "prompt_variant": "anti_refusal",
    },
]

CONFIG_MAP = {c["name"]: c for c in CONFIGS}
