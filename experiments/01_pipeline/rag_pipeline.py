"""
Step 1: RAG Response Generator
===============================
Generates AI responses using interview chunks as context.
Supports all 6 models:
  - GPT-4o-mini (OpenAI API)
  - Mistral-Small-24B (Together.ai API)
  - Qwen2.5-7B-Instruct (Together.ai API)
  - Llama-3.1-8B-Instruct (Together.ai API)
  - GEITje-7B-Ultra (HuggingFace on ALICE A100 — Dutch-specific)
  - Aya-expanse-8B (HuggingFace on ALICE A100 — multilingual)

Input:  Query + context chunks
Output: AI response + metadata

HuggingFace models require:
  pip install transformers accelerate bitsandbytes sentencepiece
  (and an NVIDIA GPU with ≥16GB VRAM)

API models require: OPENAI_API_KEY or TOGETHER_API_KEY in .env
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"

def _load_env_file():
    """Read key=value pairs directly from .env without relying on dotenv."""
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

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ── System prompt templates ────────────────────────────────────────────────────

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
}

# Default prompt style
SYSTEM_PROMPT = _SYSTEM_PROMPTS["detailed"]

# ── API model routing ──────────────────────────────────────────────────────────
# Maps model name prefix → (base_url, api_key_env_or_None, canonical_model_name)

_API_MODEL_ROUTES = {
    "gpt-4o-mini":   (None,                           "OPENAI_API_KEY",   "gpt-4o-mini"),
    "gpt":           (None,                           "OPENAI_API_KEY",   "gpt-4o-mini"),
    "mistral":       ("https://api.together.xyz/v1",  "TOGETHER_API_KEY", "mistralai/Mistral-Small-24B-Instruct-2501"),
    "qwen":          ("https://api.together.xyz/v1",  "TOGETHER_API_KEY", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    "llama":         ("https://api.together.xyz/v1",  "TOGETHER_API_KEY", "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    # Ollama local alternatives — run: ollama pull qwen2.5:7b && ollama pull mistral:7b
    "qwen-local":    ("http://localhost:11434/v1",    None,               "qwen2.5:7b"),
    "mistral-local": ("http://localhost:11434/v1",    None,               "mistral:7b"),
}

# HuggingFace models that run locally on GPU (GEITje and Aya23 only — not on Together.ai)
_HF_MODEL_IDS = {
    "geitje":  "BramVanroy/GEITje-7B-ultra",
    "aya23":   "CohereForAI/aya-expanse-8b",
}

# Cache for loaded HF models (avoid reloading between calls in same process)
_hf_cache: dict = {}  # key → (tokenizer, model)


def _is_hf_model(model: str) -> bool:
    """Return True if this model runs via HuggingFace transformers."""
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
    # Default: OpenAI
    client = openai.OpenAI()
    return client, model


def _get_hf_model(model_key: str):
    """Load (or retrieve cached) HuggingFace tokenizer + model."""
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "Run: pip install transformers accelerate sentencepiece"
        )
    if model_key not in _hf_cache:
        hf_id = _HF_MODEL_IDS[model_key]
        print(f"  Loading HuggingFace model: {hf_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
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
    """Generate a single response using a HuggingFace model."""
    tokenizer, model = _get_hf_model(model_key)
    device = next(model.parameters()).device

    # Use chat template if available, otherwise build plain prompt
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


def generate_response(
    query: str,
    context: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    system_prompt_style: str = "detailed",
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> Dict:
    """Generate one RAG response. Supports both API and HuggingFace models."""
    prompt_template = _SYSTEM_PROMPTS.get(system_prompt_style, _SYSTEM_PROMPTS["detailed"])
    system_content = prompt_template.format(context=context)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query},
    ]

    start = time.time()
    model_lower = model.lower()

    if _is_hf_model(model_lower):
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


def generate_all_responses(samples: List[Dict],
                           model: str = "gpt-4o-mini",
                           include_selfcheck_samples: bool = True,
                           n_selfcheck_samples: int = 5,
                           results_dir: str = "results/gpt-4o-mini") -> List[Dict]:
    """
    Generate RAG responses for all samples and save to results_dir/01_rag_responses.json.

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
            rag = generate_response(sample["query"], sample["context"], model=model)
            result = {
                **sample,
                "rag_response": rag["response"],
                "rag_model": model,
                "rag_tokens": rag["tokens"],
                "rag_latency": rag["latency"],
            }

            if include_selfcheck_samples:
                sampled = generate_samples(
                    sample["query"], sample["context"],
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
