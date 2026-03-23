"""
Model Registry — Central configuration for all 6 models.
=========================================================
Single source of truth for model IDs, API types, generation parameters,
and ALICE HPC resource requirements. Import MODELS anywhere in the pipeline.

Usage:
  from experiments.01_pipeline.model_registry import MODELS, get_model
  cfg = get_model("geitje")
  print(cfg["hf_id"])   # BramVanroy/GEITje-7B-ultra
"""

from __future__ import annotations

MODELS: dict[str, dict] = {

    # ── Commercial API models (no GPU needed) ──────────────────────────────

    "gpt-4o-mini": {
        "display_name":  "GPT-4o-mini",
        "type":          "openai_api",
        "api_model_id":  "gpt-4o-mini",
        "hf_id":         None,
        "size_params":   "~8B equivalent",
        "context_len":   128_000,
        "language":      "multilingual",
        "gen_params": {
            "temperature":        0.7,
            "max_tokens":         512,
            "top_p":              1.0,
        },
        "alice_gpu":     None,
        "alice_ram_gb":  None,
        "notes": (
            "Commercial baseline. Exposes safety-refusal pattern "
            "(REFUSAL_HALLUCINATION ~60% of GPT-4o-mini hallucinations). "
            "Cheapest commercial model; ~$0.15/1M input tokens."
        ),
    },

    "mistral": {
        "display_name":  "Mistral-Small-24B",
        "type":          "mistral_api",
        "api_model_id":  "mistral-small-latest",
        "hf_id":         None,
        "size_params":   "24B",
        "context_len":   32_000,
        "language":      "multilingual",
        "gen_params": {
            "temperature":        0.7,
            "max_tokens":         512,
            "top_p":              1.0,
        },
        "alice_gpu":     None,
        "alice_ram_gb":  None,
        "notes": (
            "Best-performing commercial model in our study "
            "(17.54% hallucination vs GPT-4o-mini 20.77%; "
            "McNemar p=0.0001). Strong baseline for comparison."
        ),
    },

    # ── Open HuggingFace models (require ALICE A100) ───────────────────────

    "qwen": {
        "display_name":  "Qwen2.5-7B-Instruct",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "Qwen/Qwen2.5-7B-Instruct",
        "size_params":   "7B",
        "context_len":   32_768,
        "language":      "multilingual (Chinese-developed)",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "alice_gpu":     "A100-40GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "Chinese-developed multilingual model. Highest hallucination rate "
            "(22.15%). Important for cross-cultural LLM comparison. "
            "Runs on A100 40GB in float16 (~14GB VRAM)."
        ),
    },

    "llama": {
        "display_name":  "Llama-3.1-8B-Instruct",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "meta-llama/Llama-3.1-8B-Instruct",
        "size_params":   "8B",
        "context_len":   128_000,
        "language":      "multilingual",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "alice_gpu":     "A100-40GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "Meta standard open model; most widely cited open-source LLM. "
            "128K context window is a major advantage for long interviews. "
            "Runs on A100 40GB (~16GB VRAM in float16). "
            "Required for reproducibility — reviewers expect Llama comparison."
        ),
    },

    "geitje": {
        "display_name":  "GEITje-7B-Ultra",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "BramVanroy/GEITje-7B-ultra",
        "size_params":   "7B",
        "context_len":   4_096,
        "language":      "Dutch (fine-tuned on Dutch data)",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "alice_gpu":     "A100-40GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "UNIQUE DUTCH CONTRIBUTION. Best Dutch-specific generative LLM "
            "publicly available. Mistral-7B fine-tuned on curated Dutch corpora "
            "by BramVanroy (Utrecht). Enables first-ever Dutch-specific RAG "
            "hallucination study. Expect lower hallucination on Dutch interviews "
            "vs multilingual models (~14GB VRAM in float16)."
        ),
    },

    "aya23": {
        "display_name":  "Aya-23-8B",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "CohereForAI/aya-23-8B",
        "size_params":   "8B",
        "context_len":   8_192,
        "language":      "multilingual (23 languages incl. Dutch)",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "alice_gpu":     "A100-40GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "Cohere multilingual model covering 23 languages including Dutch. "
            "Paired with GEITje to compare Dutch-specific vs multilingual "
            "approaches — key cross-lingual finding for the thesis. "
            "Command R architecture, strong instruction following. "
            "(~16GB VRAM in float16)."
        ),
    },
}

# Ordered list for consistent reporting
MODEL_ORDER = ["gpt-4o-mini", "mistral", "qwen", "llama", "geitje", "aya23"]

# Convenience groups
API_MODELS = [k for k, v in MODELS.items() if v["type"] in ("openai_api", "mistral_api")]
HF_MODELS  = [k for k, v in MODELS.items() if v["type"] == "huggingface"]
DUTCH_MODELS = ["geitje"]
MULTILINGUAL_WITH_DUTCH = ["aya23", "qwen", "llama", "mistral", "gpt-4o-mini"]


def get_model(name: str) -> dict:
    """Return model config dict; raises KeyError if unknown."""
    if name not in MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODELS.keys())}"
        )
    return MODELS[name]


def results_dir_for(model_name: str, results_base: str = "results") -> str:
    """Return the per-model results subdirectory path."""
    from pathlib import Path
    safe = model_name.replace("/", "-").replace(".", "-")
    return str(Path(results_base) / safe)


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("MODEL REGISTRY — 6-Model Lineup")
    print(f"{'='*70}")
    print(f"{'Key':<12} {'Display Name':<26} {'Type':<16} {'Size':<8} {'Context':<10} {'Lang'}")
    print("-" * 90)
    for key in MODEL_ORDER:
        m = MODELS[key]
        lang = m["language"][:25]
        print(
            f"{key:<12} {m['display_name']:<26} {m['type']:<16} "
            f"{m['size_params']:<8} {m['context_len']:<10,} {lang}"
        )
    print(f"\nAPI models (no GPU): {API_MODELS}")
    print(f"HuggingFace models:  {HF_MODELS}")
    print(f"Dutch-specific:      {DUTCH_MODELS}")
