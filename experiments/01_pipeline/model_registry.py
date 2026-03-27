"""
Model Registry — Central configuration for all 9 models.
=========================================================
Single source of truth for model IDs, API types, generation parameters,
and ALICE HPC resource requirements. Import MODELS anywhere in the pipeline.

CHANGES FROM v1 (Mistral-Small-24B era):
  - "mistral"     : was mistral_api (Together.ai, 24B) → now HuggingFace Mistral-7B-Instruct-v0.3
                    Why: Supervisor requires size parity with other 7-8B models.
                    No API token needed — free HuggingFace download.
                    Paper: Jiang et al. 2023 (arXiv:2310.06825) Table 1 p.2
  - "qwen14b"     : NEW — Qwen2.5-14B-Instruct (scale ablation: does 14B < 7B hallucination?)
                    Paper: Hui et al. 2024 (arXiv:2412.15115) Table 2 p.5
  - "mixtral"     : NEW — Mixtral-8×7B-Instruct (MoE architecture test)
                    Paper: Jiang et al. 2024 (arXiv:2401.04088)
  - "llama"       : REMOVED — HuggingFace access not granted yet
  - alice_gpu     : updated A100-40GB → A100-80GB (confirmed by cluster admin)
  - "mistral_base": NEW v3 — Mistral-7B-v0.1 (raw base, NO instruction tuning)
                    Dutch chain step 1 (Bram Vanroy recommendation)
  - "geitje_sft"  : NEW v3 — GEITje-7B-ultra-sft (SFT checkpoint before DPO)
                    Dutch chain step 2 (Bram Vanroy recommendation)

─────────────────────────────────────────────────────────────────────────────
NOTE (from Bram Vanroy, GEITje author, personal communication March 2026):
  GEITje is based on Mistral-7B-v0.1 (released 2023). Comparing it to
  modern models (Qwen2.5, Mixtral, GPT-4o-mini) is NOT a fair capability
  comparison — the field has evolved significantly since 2023.
  The correct use of GEITje in this thesis is as a DUTCH-LANGUAGE SPECIALIST
  baseline, not as a capability-matched competitor.

  The training-stage chain comparison is the PRIMARY NOVEL CONTRIBUTION:
    mistral_base (Mistral-7B-v0.1, no fine-tuning)
    → geitje_sft  (GEITje-7B-ultra-sft, Dutch SFT applied)
    → geitje      (GEITje-7B-ultra, DPO applied on top of SFT)

  Research question: "At which training stage do Dutch hallucinations emerge?
  Was it already in pretraining, did SFT exacerbate it, or did DPO introduce it?"

  Paper backing:
    - Bai et al. 2022 (arXiv:2204.05862) — RLHF training stages
    - Ouyang et al. 2022 (arXiv:2203.02155) — SFT vs RLHF comparison
─────────────────────────────────────────────────────────────────────────────

Usage:
  from experiments.model_registry import MODELS, get_model
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
        "load_in_4bit":  False,
        "alice_gpu":     None,
        "alice_ram_gb":  None,
        "alice_cpus":    None,
        "alice_time":    None,
        "notes": (
            "Commercial baseline. Exposes REFUSAL_HALLUCINATION pattern "
            "(59.5% of GPT-4o-mini hallucinations are refusals). "
            "Cheapest commercial model; ~$0.15/1M input tokens."
        ),
    },

    # ── Open HuggingFace models (require ALICE A100-80GB) ─────────────────
    # NOTE: Llama-3.1-8B removed — HuggingFace access not yet granted.
    # Re-add "llama": {...} once permission is approved.

    "mistral": {
        # CHANGED v2: was mistral_api / Together.ai / Mistral-Small-24B
        # NOW: HuggingFace Mistral-7B-Instruct-v0.3 (free, no gating)
        # Reason: supervisor requires size parity with other 7-8B open models.
        # Architecture: GQA + SWA — beats Llama-2-13B at 7B (Jiang et al. 2023, Table 1 p.2)
        "display_name":  "Mistral-7B-Instruct-v0.3",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "mistralai/Mistral-7B-Instruct-v0.3",
        "size_params":   "7B",
        "context_len":   32_768,
        "language":      "multilingual",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "load_in_4bit":  False,
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "CHANGED v2: replaced Mistral-Small-24B (API) with Mistral-7B-Instruct-v0.3 (HF). "
            "Free, no API token, no HuggingFace gating. "
            "Grouped Query Attention (GQA) + Sliding Window Attention (SWA) "
            "make 7B competitive with 13B+ models. ~14GB VRAM in float16."
        ),
    },

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
        "load_in_4bit":  False,
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "Chinese-developed multilingual model. Baseline at 7B tier. "
            "Compared with qwen14b for scale ablation. ~14GB VRAM in float16."
        ),
    },

    "qwen14b": {
        # NEW v2: Scale ablation — does doubling from 7B to 14B reduce hallucination?
        # MMLU: 79.7% (14B) vs 74.2% (7B) — Hui et al. 2024 (arXiv:2412.15115) Table 2 p.5
        # 4-bit quantization: ~10-11GB VRAM — fits A100-80GB comfortably
        "display_name":  "Qwen2.5-14B-Instruct",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "Qwen/Qwen2.5-14B-Instruct",
        "size_params":   "14B",
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
        "load_in_4bit":  True,   # 4-bit quantization to fit on A100-80GB
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "16:00:00",
        "notes": (
            "NEW v2: Scale ablation vs Qwen2.5-7B. Tests whether doubling parameters "
            "within same family reduces hallucination (McNemar test will confirm). "
            "MMLU +5.5pp over 7B variant. ~10-11GB VRAM at 4bit."
        ),
    },

    # ── Dutch training-stage chain (Bram Vanroy recommendation, March 2026) ──
    # Chain: mistral_base → geitje_sft → geitje
    # Reveals at which stage Dutch hallucinations are introduced.

    "mistral_base": {
        # Dutch chain step 1: RAW BASE MODEL — no instruction tuning, no Dutch fine-tuning.
        # Used to establish a hallucination baseline BEFORE any fine-tuning.
        # Bram Vanroy (GEITje author) specifically recommends this comparison.
        # Paper: Jiang et al. 2023 (arXiv:2310.06825) — original Mistral architecture
        "display_name":  "Mistral-7B-v0.1 (base)",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "mistralai/Mistral-7B-v0.1",
        "size_params":   "7B",
        "context_len":   32_768,
        "language":      "multilingual (no instruction tuning)",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "load_in_4bit":  False,
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "DUTCH CHAIN STEP 1. Raw base model — NO instruction tuning, NO Dutch fine-tuning. "
            "Establishes hallucination baseline before any fine-tuning. "
            "Recommended by Bram Vanroy (GEITje author, personal communication March 2026). "
            "~14GB VRAM in float16."
        ),
    },

    "geitje_sft": {
        # Dutch chain step 2: SFT CHECKPOINT — Dutch supervised fine-tuning applied,
        # but NOT yet DPO. Tests whether SFT alone increases or decreases hallucination.
        # Paper: Ouyang et al. 2022 (arXiv:2203.02155) — SFT stage of RLHF pipeline
        "display_name":  "GEITje-7B-ultra-sft",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "BramVanroy/GEITje-7B-ultra-sft",
        "size_params":   "7B",
        "context_len":   4_096,
        "language":      "Dutch (SFT applied, before DPO)",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "load_in_4bit":  False,
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "DUTCH CHAIN STEP 2. SFT checkpoint BEFORE DPO alignment. "
            "Tests whether supervised Dutch fine-tuning alone raises/lowers hallucination vs mistral_base. "
            "Recommended by Bram Vanroy (GEITje author, personal communication March 2026). "
            "~14GB VRAM in float16."
        ),
    },

    "geitje": {
        "display_name":  "GEITje-7B-Ultra (DPO)",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "BramVanroy/GEITje-7B-ultra",
        "size_params":   "7B",
        "context_len":   4_096,
        "language":      "Dutch (SFT + DPO fine-tuned)",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "load_in_4bit":  False,
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "DUTCH CHAIN STEP 3 (final DPO). Best Dutch-specific generative LLM publicly available. "
            "Mistral-7B-v0.1 + Dutch SFT + DPO by BramVanroy (Utrecht University). "
            "HIGHEST measured hallucination rate: 65.9% (2048/3108 responses). "
            "QLoRA fine-tuning target: rank=16 on Q/K/V/O layers. ~14GB VRAM in float16. "
            "Recommended by Bram Vanroy for training-stage chain comparison (March 2026)."
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
        "load_in_4bit":  False,
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  32,
        "alice_cpus":    4,
        "alice_time":    "12:00:00",
        "notes": (
            "Cohere multilingual model covering 23 languages including Dutch. "
            "Paired with GEITje: multilingual vs Dutch-specialist comparison. "
            "Command R architecture, strong instruction following. ~16GB VRAM in float16."
        ),
    },

    "mixtral": {
        # NEW v2: Mixture-of-Experts architecture test
        # Only 2 of 8 expert networks activate per token → expert routing separates domain knowledge
        # TruthfulQA: 71.4% (Jiang et al. 2024, arXiv:2401.04088, Table 3 p.5)
        # Hypothesis: MoE routing may reduce ROLE_ATTRIBUTION_DRIFT by keeping
        # speaker-specific knowledge more separated in expert networks
        # 4-bit quantization: ~24-28GB VRAM → fits A100-80GB
        "display_name":  "Mixtral-8x7B-Instruct-v0.1",
        "type":          "huggingface",
        "api_model_id":  None,
        "hf_id":         "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "size_params":   "46.7B total / 12.9B active",
        "context_len":   32_768,
        "language":      "multilingual",
        "gen_params": {
            "temperature":        0.7,
            "max_new_tokens":     512,
            "top_p":              0.9,
            "repetition_penalty": 1.1,
            "do_sample":          True,
        },
        "torch_dtype":   "float16",
        "load_in_4bit":  True,   # 4-bit: ~24-28GB VRAM — fits A100-80GB
        "alice_gpu":     "A100-80GB",
        "alice_ram_gb":  48,     # Higher RAM needed for MoE routing
        "alice_cpus":    4,
        "alice_time":    "16:00:00",
        "notes": (
            "NEW v2: Mixture-of-Experts architecture test. "
            "Only 12.9B of 46.7B parameters active per forward pass. "
            "Tests whether MoE expert separation reduces ROLE_ATTRIBUTION_DRIFT. "
            "~24-28GB VRAM at 4bit — fits A100-80GB."
        ),
    },
}

# ── Ordered model list for consistent reporting ────────────────────────────
# Note: llama removed (no HuggingFace access yet)
# Dutch training-stage chain: mistral_base → geitje_sft → geitje (Bram Vanroy, March 2026)
MODEL_ORDER = [
    "gpt-4o-mini",
    "mistral_base", "mistral",          # Mistral: base → instruction-tuned
    "qwen", "qwen14b",                  # Qwen: 7B → 14B scale ablation
    "geitje_sft", "geitje",             # Dutch chain: SFT → DPO
    "aya23", "mixtral",                 # Multilingual + MoE
]

# Convenience groups
API_MODELS  = [k for k, v in MODELS.items() if v["type"] == "openai_api"]
HF_MODELS   = [k for k, v in MODELS.items() if v["type"] == "huggingface"]
HF_4BIT     = [k for k, v in MODELS.items() if v.get("load_in_4bit")]
DUTCH_MODELS        = ["mistral_base", "geitje_sft", "geitje"]  # Dutch training-stage chain
DUTCH_CHAIN         = ["mistral_base", "geitje_sft", "geitje"]  # alias for clarity
MULTILINGUAL_WITH_DUTCH = ["aya23", "qwen", "qwen14b", "mistral", "gpt-4o-mini", "mixtral"]


def get_model(name: str) -> dict:
    """Return model config dict; raises KeyError with helpful message if unknown."""
    if name not in MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODELS.keys())}\n"
            f"Note: 'llama' was removed pending HuggingFace access approval."
        )
    return MODELS[name]


def results_dir_for(model_name: str, results_base: str = "results") -> str:
    """Return the per-model results subdirectory path."""
    from pathlib import Path
    safe = model_name.replace("/", "-").replace(".", "-")
    return str(Path(results_base) / safe)


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("MODEL REGISTRY — 9-Model Lineup (v3: +mistral_base, +geitje_sft, Dutch chain)")
    print(f"{'='*80}")
    print(f"{'Key':<12} {'Display Name':<30} {'Type':<16} {'Size':<18} {'4bit':<6} {'GPU'}")
    print("-" * 95)
    for key in MODEL_ORDER:
        m = MODELS[key]
        gpu = m["alice_gpu"] or "API (no GPU)"
        fourbit = "yes" if m.get("load_in_4bit") else "-"
        print(
            f"{key:<12} {m['display_name']:<30} {m['type']:<16} "
            f"{m['size_params']:<18} {fourbit:<6} {gpu}"
        )
    print(f"\nAPI models (no GPU)  : {API_MODELS}")
    print(f"HuggingFace models   : {HF_MODELS}")
    print(f"4-bit quantized      : {HF_4BIT}")
    print(f"Dutch-specialist     : {DUTCH_MODELS}")
    print(f"\nLlama: REMOVED — add back once HuggingFace access approved.")
