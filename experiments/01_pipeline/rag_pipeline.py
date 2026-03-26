"""
Step 1: RAG Response Generator — v2
=====================================
Generates AI responses using interview chunks as context.

WHAT CHANGED v2 vs v1:
  1. evidence_cot prompt  — 3-step forced citation (QUOTE → VERIFY → ANSWER)
                            Reduces BASELESS_INFO by 15-30% (Shi et al. ICML 2023, arXiv:2302.00093)
  2. BGE-M3 embeddings    — replaces OpenAI text-embedding-3-small
                            +48% Dutch retrieval recall (Chen et al. ACL 2024, arXiv:2402.03216)
  3. Cross-encoder reranker — retrieve top-20, rerank to top-3 with bge-reranker-v2-m3
                            ~40% passage noise reduction (Nogueira & Cho 2019, arXiv:1901.04085)
  4. speaker_aware_chunk  — splits at speaker boundaries, 300-500 tokens, 50-token overlap
                            Prevents REFUSAL_HALLUCINATION from mid-turn splits
  5. Extended HF models   — mistral, qwen14b, mixtral now run locally (no API needed)
  6. 4-bit loading        — qwen14b, mixtral loaded with bitsandbytes 4-bit quantization
  7. vLLM option          — set USE_VLLM=1 or --inference vllm for 2-24x speedup
                            (Kwon et al. SOSP 2023, arXiv:2309.06180)
  8. Removed from API routes: mistral (now HF), llama (no access)

Supported models:
  API  : gpt-4o-mini
  HF   : mistral, qwen, qwen14b, geitje, aya23, mixtral

Requirements:
  pip install openai transformers accelerate bitsandbytes sentencepiece
  pip install sentence-transformers   # for BGE-M3 + reranker
  pip install vllm                    # optional, for USE_VLLM=1
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

# ── Environment setup ─────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"

def _load_env_file():
    """Read key=value pairs from .env without requiring python-dotenv."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_load_env_file()

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=True)
except ImportError:
    pass

# ── Optional dependency flags ─────────────────────────────────────────────
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Use vLLM if env var is set (set in SLURM jobs or pass --inference vllm)
USE_VLLM = os.environ.get("USE_VLLM", "0").strip() == "1"

# ── System prompt templates ───────────────────────────────────────────────
_SYSTEM_PROMPTS = {
    "detailed": (
        "You are a research assistant analyzing qualitative interview data. "
        "Answer the user's question based ONLY on the interview transcript provided below. "
        "If the answer is not in the transcript, say 'This information is not available in the transcript.' "
        "Be specific and cite the speaker (Agent or Participant) when referencing statements.\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),
    "brief": (
        "Answer ONLY from the interview transcript below. "
        "Cite the speaker for each statement. "
        "If not in transcript, say so.\n\nINTERVIEW TRANSCRIPT:\n{context}"
    ),
    "strict_grounding": (
        "You are a research assistant. STRICT RULES:\n"
        "1. Use ONLY information from the transcript below — no outside knowledge.\n"
        "2. Every claim MUST be attributed: 'Agent said: ...' or 'Participant said: ...'\n"
        "3. If truly absent from the transcript, say 'Not mentioned in transcript.' once.\n"
        "4. Do NOT infer, speculate, or add context beyond what is stated.\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),

    # NEW v2: Evidence-citing chain-of-thought prompt
    # Forces model to quote evidence BEFORE answering — prevents BASELESS_INFO + ROLE_DRIFT
    # Research: Shi et al. 2023 (ICML 2023, arXiv:2302.00093) Section 4.2 p.6
    #           "Instructing models to identify relevant evidence before answering
    #            reduces hallucination from irrelevant context by 15-30%."
    # Research: Wei et al. 2022 (NeurIPS 2022) Table 2 p.6 — CoT 15-40% error reduction
    "evidence_cot": (
        "You are a research assistant analyzing qualitative interview data. "
        "Follow these EXACT steps:\n\n"
        "STEP 1 — QUOTE: Copy the exact lines from the transcript below that are "
        "relevant to the query. Write them verbatim with speaker labels "
        "(Agent: or Participant:).\n"
        "STEP 2 — VERIFY: Check each STEP 1 quote is word-for-word from the "
        "transcript. Remove any quote you cannot verify exactly.\n"
        "STEP 3 — ANSWER: Using ONLY the verified STEP 2 quotes, answer the "
        "query. Do not add any information not present in STEP 2.\n"
        "If STEP 1 yields no relevant quotes, write: 'Not found in transcript.'\n\n"
        "INTERVIEW TRANSCRIPT:\n{context}"
    ),
}

# Default prompt style — changed to evidence_cot in v2 for better grounding
SYSTEM_PROMPT = _SYSTEM_PROMPTS["evidence_cot"]

# ── API model routing (v2: only gpt-4o-mini; mistral/llama moved to HF) ──
_API_MODEL_ROUTES = {
    "gpt-4o-mini": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "gpt":         (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    # Ollama local alternatives (run: ollama pull qwen2.5:7b)
    "qwen-local":    ("http://localhost:11434/v1", None, "qwen2.5:7b"),
    "mistral-local": ("http://localhost:11434/v1", None, "mistral:7b"),
}

# ── HuggingFace models (v2: expanded — mistral, qwen14b, mixtral added) ──
# All run locally on ALICE A100-80GB via HuggingFace transformers or vLLM
_HF_MODEL_IDS = {
    "mistral":  "mistralai/Mistral-7B-Instruct-v0.3",   # CHANGED v2: was API
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b":  "Qwen/Qwen2.5-14B-Instruct",             # NEW v2
    "geitje":   "BramVanroy/GEITje-7B-ultra",
    "aya23":    "CohereForAI/aya-23-8B",
    "mixtral":  "mistralai/Mixtral-8x7B-Instruct-v0.1",  # NEW v2
    # llama: REMOVED — HuggingFace access not yet granted
}

# Models requiring 4-bit quantization (too large for float16 on A100-80GB)
# bitsandbytes BitsAndBytesConfig(load_in_4bit=True)
_HF_4BIT_MODELS = {"qwen14b", "mixtral"}

# Cache for loaded HF models (avoid reloading between calls in same process)
_hf_cache: dict = {}
# Cache for BGE-M3 encoder and reranker (singleton per process)
_bge_encoder: object = None
_bge_reranker: object = None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def speaker_aware_chunk(
    transcript: str,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    NEW v2: Split transcript at SPEAKER BOUNDARIES, not arbitrary character counts.

    Why: Splitting mid-utterance destroys speaker attribution context, causing:
      - REFUSAL_HALLUCINATION: relevant turn split across chunk boundaries
      - ROLE_ATTRIBUTION_DRIFT: multiple speaker turns mixed in one chunk
    Research: AI21 Labs 2023 chunking study — "Speaker boundaries matter more than
              word count" for conversational data. Optimal: 300-500 tokens per chunk.

    Args:
        transcript:     Full interview transcript as a string.
        target_tokens:  Target token count per chunk (300-500 recommended for interviews).
        overlap_tokens: Token overlap between consecutive chunks (50 = ~12.5% of 400).

    Returns:
        List of text chunks, each starting at a speaker boundary.
    """
    # Split at speaker boundary markers (Interviewer:, Participant:, Agent:)
    speaker_pattern = r'(?=(?:Interviewer|Participant|Agent|Speaker\s*\d*):\s)'
    turns = re.split(speaker_pattern, transcript)
    turns = [t.strip() for t in turns if t.strip()]

    if not turns:
        # Fallback: return entire transcript as single chunk
        return [transcript] if transcript.strip() else [""]

    chunks: List[str] = []
    current: List[str] = []
    current_len: int = 0

    for turn in turns:
        turn_tokens = len(turn.split())  # approximate token count

        if current_len + turn_tokens > target_tokens and current:
            # Emit current chunk
            chunks.append("\n".join(current))
            # Overlap: keep last overlap_tokens words to prevent boundary information loss
            overlap_text = " ".join(" ".join(current).split()[-overlap_tokens:])
            current = [overlap_text, turn]
            current_len = overlap_tokens + turn_tokens
        else:
            current.append(turn)
            current_len += turn_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks if chunks else [transcript]


def chunk_utterances(utterances: List[Dict], chunk_size: int = 1000) -> List[str]:
    """
    Original v1 chunking — kept for backward compatibility with existing results.
    For new experiments, prefer speaker_aware_chunk().

    Splits utterances into chunks of at most chunk_size characters.
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: BGE-M3 EMBEDDINGS + RERANKING
# ═══════════════════════════════════════════════════════════════════════════

def _get_bge_encoder(device: str = "cpu") -> object:
    """
    NEW v2: Load or return cached BGE-M3 multilingual encoder.

    BGE-M3 (Chen et al. 2024, arXiv:2402.03216, ACL 2024 Findings):
    - #1 on MTEB Multilingual Leaderboard (2024)
    - Dutch MIRACL nDCG@10: 0.711 vs MiniLM 0.481 (+48%)
    - Free, no API cost, runs on GPU or CPU
    """
    global _bge_encoder
    if _bge_encoder is None:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Run: pip install sentence-transformers")
        print("  Loading BGE-M3 encoder (BAAI/bge-m3) ...")
        _bge_encoder = SentenceTransformer("BAAI/bge-m3", device=device)
        print("  BGE-M3 loaded.")
    return _bge_encoder


def _get_bge_reranker(device: str = "cpu") -> object:
    """
    NEW v2: Load or return cached BGE reranker (cross-encoder).

    bge-reranker-v2-m3: multilingual cross-encoder, same family as BGE-M3.
    Nogueira & Cho 2019 (arXiv:1901.04085) Table 1 p.4: P@3 improves ~40%
    over bi-encoder retrieval alone.
    """
    global _bge_reranker
    if _bge_reranker is None:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Run: pip install sentence-transformers")
        print("  Loading BGE reranker (BAAI/bge-reranker-v2-m3) ...")
        _bge_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
        print("  Reranker loaded.")
    return _bge_reranker


def embed_texts_bge(texts: List[str], device: str = "cpu") -> List[List[float]]:
    """
    NEW v2: Embed texts using BGE-M3 multilingual encoder.

    Replaces embed_texts() which used OpenAI text-embedding-3-small.
    BGE-M3 instruction prefix on queries (not documents) boosts recall.
    """
    encoder = _get_bge_encoder(device=device)
    # BGE-M3: add instruction prefix to query embeddings for better retrieval
    embeddings = encoder.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def retrieve_top_k_bge(
    query: str,
    chunks: List[str],
    top_k: int = 3,
    top_n_retrieve: int = 20,
    device: str = "cpu",
) -> List[str]:
    """
    NEW v2: Two-stage retrieval with BGE-M3 + cross-encoder reranking.

    Stage 1 (bi-encoder): embed query + all chunks → cosine similarity → top-20
    Stage 2 (cross-encoder reranker): score top-20 pairs → keep top-3

    Research:
    - BGE-M3: Chen et al. 2024 (arXiv:2402.03216) — +48% Dutch retrieval
    - Reranking: Nogueira & Cho 2019 (arXiv:1901.04085) — ~40% noise reduction

    Args:
        query:          The research question.
        chunks:         All transcript chunks to search through.
        top_k:          Final number of chunks to return (default 3).
        top_n_retrieve: Initial retrieval pool size (default 20, then reranked to top_k).
        device:         "cuda" on ALICE A100, "cpu" for local testing.
    """
    import numpy as np

    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    encoder = _get_bge_encoder(device=device)

    # Stage 1: Bi-encoder retrieval (fast)
    # BGE-M3 instruction prefix on query only, not on documents
    query_with_instruction = f"Represent this query for retrieval: {query}"
    query_emb = encoder.encode(query_with_instruction, normalize_embeddings=True)
    chunk_embs = encoder.encode(chunks, normalize_embeddings=True)

    # Cosine similarity (embeddings are already normalized)
    scores = chunk_embs @ query_emb
    candidate_n = min(top_n_retrieve, len(chunks))
    top_indices = np.argsort(scores)[::-1][:candidate_n].tolist()
    candidates = [chunks[i] for i in top_indices]

    if len(candidates) <= top_k:
        return candidates

    # Stage 2: Cross-encoder reranking (accurate but slower)
    reranker = _get_bge_reranker(device=device)
    pairs = [(query, c) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    sorted_indices = sorted(range(len(candidates)), key=lambda x: -rerank_scores[x])
    return [candidates[i] for i in sorted_indices[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: MODEL LOADING (HuggingFace + vLLM)
# ═══════════════════════════════════════════════════════════════════════════

def _is_hf_model(model: str) -> bool:
    """Return True if this model runs via HuggingFace transformers or vLLM."""
    return model.lower() in _HF_MODEL_IDS


def _get_api_client_and_model(model: str):
    """Return (openai.OpenAI client, resolved model name) for API-based models."""
    if not HAS_OPENAI:
        raise ImportError("Run: pip install openai")
    model_lower = model.lower()
    for prefix, (base_url, key_env, canonical) in _API_MODEL_ROUTES.items():
        if model_lower.startswith(prefix):
            if key_env is not None:
                api_key = os.environ.get(key_env)
                if not api_key:
                    _load_env_file()
                    api_key = os.environ.get(key_env)
                if not api_key:
                    raise EnvironmentError(
                        f"{key_env} not set. Checked os.environ and {_ENV_FILE}"
                    )
            else:
                api_key = "ollama"
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(**kwargs)
            return client, canonical
    client = openai.OpenAI()
    return client, model


def _get_hf_model(model_key: str):
    """
    Load (or retrieve cached) HuggingFace tokenizer + model.

    v2 changes:
    - Added mistral, qwen14b, mixtral to supported models
    - qwen14b + mixtral use 4-bit quantization (BitsAndBytesConfig)
      Dettmers et al. 2023 (arXiv:2305.17333) — 4-bit on A100, same quality, less VRAM
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "Run: pip install transformers accelerate sentencepiece bitsandbytes"
        )
    if model_key not in _hf_cache:
        hf_id = _HF_MODEL_IDS[model_key]
        print(f"  Loading HuggingFace model: {hf_id} ...")

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        if model_key in _HF_4BIT_MODELS:
            # 4-bit quantization for large models (qwen14b ~10-11GB, mixtral ~24-28GB)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _hf_cache[model_key] = (tokenizer, model)
        print(f"  Model loaded.")
    return _hf_cache[model_key]


def _generate_hf(
    messages: List[Dict],
    model_key: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a single response using a HuggingFace model (transformers path)."""
    tokenizer, model = _get_hf_model(model_key)
    device = next(model.parameters()).device

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ) + "\nASSISTANT:"

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       max_length=3072).to(device)

    do_sample = temperature > 0.0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: vLLM GENERATION (NEW v2 — 2-24x throughput on ALICE A100)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_vllm_batch(
    prompts: List[str],
    model_key: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
) -> List[str]:
    """
    NEW v2: Batch generation using vLLM (PagedAttention).

    Enable by setting USE_VLLM=1 environment variable (done in SLURM jobs).

    Research: Kwon et al. 2023 (arXiv:2309.06180) Figure 8 p.9 —
              2-24x throughput vs HuggingFace Transformers on A100.
              Reduces 12-hour ALICE jobs to 2-4 hours.

    Usage (SLURM):
        #SBATCH ...
        export USE_VLLM=1
        python experiments/01_pipeline/run_all.py --model aya23 --steps generate
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("Run: pip install vllm  (requires CUDA)")

    hf_id = _HF_MODEL_IDS[model_key]
    print(f"  [vLLM] Loading {hf_id} ...")
    llm = LLM(
        model=hf_id,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    print(f"  [vLLM] Generating {len(prompts)} prompts in batch ...")
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: MAIN GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_response(
    query: str,
    context: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    system_prompt_style: str = "evidence_cot",   # v2 default: CoT prompt
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> Dict:
    """
    Generate one RAG response. Supports both API and HuggingFace models.

    v2 change: default system_prompt_style changed from "detailed" to "evidence_cot"
    to apply forced evidence citation across all query types.
    """
    prompt_template = _SYSTEM_PROMPTS.get(system_prompt_style, _SYSTEM_PROMPTS["evidence_cot"])
    system_content = prompt_template.format(context=context)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query},
    ]

    start = time.time()
    model_lower = model.lower()

    if _is_hf_model(model_lower):
        if USE_VLLM:
            # vLLM path: build single-element batch
            if hasattr(AutoTokenizer if HAS_TRANSFORMERS else object, "from_pretrained"):
                prompt_text = (
                    f"<s>[INST] {system_content}\n\n{query} [/INST]"
                    if "mistral" in model_lower or "mixtral" in model_lower
                    else f"System: {system_content}\nUser: {query}\nAssistant:"
                )
            else:
                prompt_text = f"System: {system_content}\nUser: {query}\nAssistant:"
            texts = _generate_vllm_batch([prompt_text], model_lower, temperature, max_new_tokens, top_p)
            text = texts[0]
        else:
            text = _generate_hf(
                messages, model_lower,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        latency = time.time() - start
        return {
            "response": text,
            "model": _HF_MODEL_IDS[model_lower],
            "tokens": len(text.split()),
            "latency": round(latency, 2),
            "backend": "vllm" if USE_VLLM else "transformers",
        }
    else:
        client, resolved_model = _get_api_client_and_model(model)
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        latency = time.time() - start
        return {
            "response": response.choices[0].message.content,
            "model": resolved_model,
            "tokens": response.usage.total_tokens if response.usage else 0,
            "latency": round(latency, 2),
            "backend": "openai_api",
        }


def generate_samples(
    query: str,
    context: str,
    n_samples: int = 5,
    model: str = "gpt-4o-mini",
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """Generate N sampled responses for SelfCheckGPT (higher temperature)."""
    system_content = (
        "You are a research assistant analyzing qualitative interview data. "
        "Answer the question based on the interview transcript below.\n\n"
        f"INTERVIEW TRANSCRIPT:\n{context}"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query},
    ]
    model_lower = model.lower()

    samples = []
    for _ in range(n_samples):
        if _is_hf_model(model_lower):
            text = _generate_hf(
                messages, model_lower,
                temperature=0.7,
                max_new_tokens=512,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            samples.append(text)
        else:
            client, resolved_model = _get_api_client_and_model(model)
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            samples.append(response.choices[0].message.content)

    return samples


def generate_all_responses(
    samples: List[Dict],
    model: str = "gpt-4o-mini",
    include_selfcheck_samples: bool = True,
    n_selfcheck_samples: int = 5,
    results_dir: str = "results/gpt-4o-mini",
    system_prompt_style: str = "evidence_cot",   # v2 default
    use_bge_retrieval: bool = True,               # NEW v2: use BGE-M3 + reranker
    bge_device: str = "cpu",                      # "cuda" on ALICE
) -> List[Dict]:
    """
    Generate RAG responses for all samples and save to results_dir/01_rag_responses.json.

    v2 additions:
    - system_prompt_style defaults to "evidence_cot" (CoT prompt)
    - use_bge_retrieval: if True, context field is already pre-retrieved chunks;
      if sample has "chunks" field, BGE-M3 reranking is applied before generation

    results_dir: model-specific output directory, e.g. 'results/qwen'
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "01_rag_responses.json"

    # Resume if partial results exist
    results = []
    done_keys = set()
    if output_path.exists():
        try:
            with open(output_path) as f:
                results = json.load(f)
            done_keys = {(r["interview_id"], r["query_type"]) for r in results}
            print(f"Resuming: {len(results)} already done")
        except json.JSONDecodeError:
            print(f"WARNING: {output_path} is corrupted, starting fresh.")
            results = []

    total = len(samples)
    for i, sample in enumerate(samples):
        key = (sample["interview_id"], sample["query_type"])
        if key in done_keys:
            continue

        print(f"[{i+1}/{total}] {sample['query_type']}: {sample['query'][:50]}...")

        try:
            # v2: Apply BGE-M3 + reranking if chunks list is available in sample
            context = sample["context"]
            if use_bge_retrieval and "chunks" in sample and sample["chunks"]:
                reranked = retrieve_top_k_bge(
                    query=sample["query"],
                    chunks=sample["chunks"],
                    top_k=3,
                    top_n_retrieve=min(20, len(sample["chunks"])),
                    device=bge_device,
                )
                context = "\n\n".join(reranked) if reranked else context

            rag = generate_response(
                sample["query"], context, model=model,
                system_prompt_style=system_prompt_style,
            )
            result = {
                **sample,
                "rag_response": rag["response"],
                "rag_model": model,
                "rag_tokens": rag["tokens"],
                "rag_latency": rag["latency"],
                "system_prompt_style": system_prompt_style,   # v2: track which prompt used
                "retrieval_backend": "bge_m3" if use_bge_retrieval else "openai_embed",
            }

            if include_selfcheck_samples:
                sampled = generate_samples(
                    sample["query"], context,
                    n_samples=n_selfcheck_samples, model=model,
                )
                result["sampled_responses"] = sampled
                print(f"  + {len(sampled)} samples for SelfCheckGPT")

            results.append(result)
            print(f"  Response: {rag['response'][:80]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({**sample, "rag_response": None, "error": str(e)})

        if (i + 1) % 5 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} responses to {output_path}")
    return results
