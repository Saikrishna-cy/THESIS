"""
Step 1: RAG Response Generator
===============================
Generates AI responses using interview chunks as context.
Supports GPT-4o-mini (OpenAI), Qwen2.5-7B (Together.ai), and Mistral-Small-24B (Together.ai).

Input:  Query + context chunks
Output: AI response + metadata
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict

_ENV_FILE = Path(__file__).parent.parent / ".env"

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

SYSTEM_PROMPT = (
    "You are a research assistant analyzing qualitative interview data. "
    "Answer the user's question based ONLY on the interview transcript provided below. "
    "If the answer is not in the transcript, say 'This information is not available in the transcript.' "
    "Be specific and cite the speaker (Agent or Participant) when referencing statements.\n\n"
    "INTERVIEW TRANSCRIPT:\n{context}"
)

# Model routing: maps model name prefix → (base_url, api_key_env_or_None, canonical_model_name)
# Both Qwen and Mistral route through Together.ai — only TOGETHER_API_KEY needed.
_MODEL_ROUTES = {
    "qwen":          ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    "mistral":       ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "mistralai/Mistral-Small-24B-Instruct-2501"),
    # Ollama local alternatives — run: ollama pull qwen2.5:7b && ollama pull mistral:7b
    "qwen-local":    ("http://localhost:11434/v1",   None,               "qwen2.5:7b"),
    "mistral-local": ("http://localhost:11434/v1",   None,               "mistral:7b"),
}


def _get_client_and_model(model: str):
    """Return (openai.OpenAI client, resolved model name) based on model name."""
    if not HAS_OPENAI:
        raise ImportError("Run: pip install openai")
    model_lower = model.lower()
    for prefix, (base_url, key_env, canonical) in _MODEL_ROUTES.items():
        if model_lower.startswith(prefix):
            if key_env is not None:
                api_key = os.environ.get(key_env)
                if not api_key:
                    # Last-resort: re-read .env directly
                    _load_env_file()
                    api_key = os.environ.get(key_env)
                if not api_key:
                    raise EnvironmentError(
                        f"{key_env} not set. Checked os.environ and {_ENV_FILE}"
                    )
            else:
                api_key = "ollama"
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            return client, canonical
    # Default: OpenAI
    client = openai.OpenAI()
    return client, model


def generate_response(query: str, context: str,
                      model: str = "gpt-4o-mini",
                      temperature: float = 0.3) -> Dict:
    """Generate one RAG response."""
    client, resolved_model = _get_client_and_model(model)
    prompt = SYSTEM_PROMPT.format(context=context)

    start = time.time()
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    latency = time.time() - start

    return {
        "response": response.choices[0].message.content,
        "model": resolved_model,
        "tokens": response.usage.total_tokens if response.usage else 0,
        "latency": round(latency, 2),
    }


def generate_samples(query: str, context: str,
                     n_samples: int = 5,
                     model: str = "gpt-4o-mini") -> List[str]:
    """Generate N sampled responses for SelfCheckGPT (higher temperature)."""
    client, resolved_model = _get_client_and_model(model)
    prompt = (
        "You are a research assistant analyzing qualitative interview data. "
        "Answer the question based on the interview transcript below.\n\n"
        f"INTERVIEW TRANSCRIPT:\n{context}"
    )

    samples = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=500,
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
