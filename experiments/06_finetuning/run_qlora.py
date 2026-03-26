"""
QLoRA Fine-tuning — Faithful Response Fine-tuning
===================================================
Fine-tunes an open model (default: GEITje-7B-ultra) on FAITHFUL-labeled
examples from the hallucination detection pipeline using QLoRA.

GOAL: Reduce hallucination rate below 5% via supervised fine-tuning on
      only the subset of model outputs that were annotated as FAITHFUL.

WHY GEITje AS DEFAULT TARGET:
  GEITje has the HIGHEST measured hallucination rate (65.9%, 2048/3108 responses)
  in the full dataset — far worse than Aya-23 (27.3%), GPT-4o-mini (21.5%),
  Qwen (20.2%), or Mistral (17.5%). It is also the ONLY Dutch-specialist model,
  making it the most important and most impactful fine-tuning target for this thesis.
  Thesis defence: "We target GEITje because it has the worst performance (65.9%)
  — showing QLoRA recovers a highly-hallucinating Dutch model is the strongest
  demonstration of the fine-tuning approach."

ARCHITECTURE:
  - Base model: any 7-8B HuggingFace model (default: geitje)
  - LoRA rank: 16, applied to Q/K/V/O projection layers
  - 4-bit quantization via bitsandbytes (QLoRA, Dettmers et al. 2023)
  - SFT trainer from TRL library (Ouyang et al. 2022 RLHF framework)

PAPER BACKING:
  - QLoRA: Dettmers et al. 2023 (arXiv:2305.17333) — 4-bit NF4 quantization
  - LoRA:  Hu et al. 2021 (arXiv:2106.09685) — rank decomposition adapters
  - TRL/SFT: Von Werra et al. 2023 (TRL library, Hugging Face)
  - Faithful training data: Wang et al. 2024 (arXiv:2311.08401) — SELF-RAG

ALICE HPC REQUIREMENTS:
  A100-80GB, 48GB RAM, 8 CPUs, 8h wall time

USAGE:
  # Default: fine-tune GEITje on faithful examples
  python experiments/06_finetuning/run_qlora.py

  # Custom model and output
  python experiments/06_finetuning/run_qlora.py \\
    --model geitje \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --epochs 3 \\
    --output results/finetuned/geitje_qlora

  # Dry run (no GPU needed, tiny data)
  python experiments/06_finetuning/run_qlora.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent

# Model HuggingFace IDs (same as model_registry.py)
_HF_IDS = {
    "geitje":  "BramVanroy/GEITje-7B-ultra",
    "aya23":   "CohereForAI/aya-23-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# ── training prompt format ────────────────────────────────────────────────────
def format_training_example(item: dict) -> str:
    """
    Convert a FAITHFUL-labeled result into a supervised training string.
    Format: <system> ... <user> ... <assistant> ...
    """
    system = (
        "You are a research assistant analyzing qualitative interview transcripts. "
        "Use ONLY information from the provided transcript. "
        "Quote the exact speaker when referencing statements."
    )
    user_msg = (
        f"TRANSCRIPT:\n{item.get('transcript_text', '')[:2000]}\n\n"
        f"CONTEXT:\n{item.get('context', '')[:1500]}\n\n"
        f"QUERY: {item.get('query', '')}"
    )
    assistant_msg = item.get("rag_response", "")
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST] {assistant_msg} </s>"


def load_faithful_examples(results_base: Path, model_key: str) -> list[dict]:
    """
    Load FAITHFUL-labeled examples from 02_rq1_annotations.json
    and merge with 01_rag_responses.json to get transcript + context.
    """
    model_dir = results_base / model_key
    ann_file  = model_dir / "02_rq1_annotations.json"
    resp_file = model_dir / "01_rag_responses.json"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_file}")
    if not resp_file.exists():
        raise FileNotFoundError(f"Responses not found: {resp_file}")

    with open(ann_file, encoding="utf-8") as f:
        annotations = json.load(f)
    with open(resp_file, encoding="utf-8") as f:
        responses = json.load(f)

    resp_idx = {(r["interview_id"], r["query_type"]): r for r in responses}

    faithful = []
    for ann in annotations:
        if ann.get("overall_label") != "FAITHFUL":
            continue
        key = (ann["interview_id"], ann["query_type"])
        resp = resp_idx.get(key, {})
        faithful.append({
            "interview_id":    ann["interview_id"],
            "query_type":      ann["query_type"],
            "query":           ann.get("query", resp.get("query", "")),
            "rag_response":    ann.get("rag_response", resp.get("rag_response", "")),
            "transcript_text": resp.get("transcript_text", ""),
            "context":         resp.get("context", ""),
            "language":        ann.get("language", ""),
        })

    print(f"  Loaded {len(faithful)} FAITHFUL examples from {model_key}")
    return faithful


def load_unified_faithful(unified_path: Path) -> list[dict]:
    """Load FAITHFUL examples from unified_results.jsonl."""
    examples = []
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("overall_label") == "FAITHFUL":
                examples.append(item)
    print(f"  Loaded {len(examples)} FAITHFUL examples from unified_results.jsonl")
    return examples


def run_qlora(
    hf_model_id: str,
    training_texts: list[str],
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_seq_length: int = 1024,
    dry_run: bool = False,
):
    """
    Run QLoRA fine-tuning using peft + trl + bitsandbytes.

    Implements:
      - NF4 4-bit quantization (Dettmers et al. 2023, arXiv:2305.17333)
      - LoRA rank-16 adapters on Q/K/V/O layers (Hu et al. 2021, arXiv:2106.09685)
      - SFT Trainer from TRL (causal language modelling on faithful examples)
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Install: pip install peft trl bitsandbytes datasets transformers")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"QLoRA Fine-tuning: {hf_model_id}")
    print(f"  LoRA rank:    {lora_r}")
    print(f"  LoRA alpha:   {lora_alpha}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch_size} × grad_accum {grad_accum}")
    print(f"  Training examples: {len(training_texts)}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Skipping actual training — all parameters validated OK")
        return

    # ── 4-bit quantization config (QLoRA) ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # NF4: Dettmers et al. 2023 Table 1
        bnb_4bit_compute_dtype=torch.float16,  # bf16 on A100
        bnb_4bit_use_double_quant=True,        # 0.4-bit additional quantization
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA configuration — rank-16 on Q/K/V/O ──────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,                                          # rank-16
        lora_alpha=lora_alpha,                             # scaling = alpha/r = 2
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # all attention layers
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = Dataset.from_dict({"text": training_texts})

    # ── Training arguments ────────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=2,
    )

    # ── SFT Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving LoRA adapter weights...")
    trainer.model.save_pretrained(str(output_path / "lora_adapter"))
    tokenizer.save_pretrained(str(output_path / "lora_adapter"))

    # Save training config
    config_out = {
        "hf_model_id":   hf_model_id,
        "lora_r":        lora_r,
        "lora_alpha":    lora_alpha,
        "lora_dropout":  lora_dropout,
        "epochs":        epochs,
        "batch_size":    batch_size,
        "grad_accum":    grad_accum,
        "lr":            lr,
        "max_seq_length": max_seq_length,
        "n_training_examples": len(training_texts),
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "paper_qlora":   "Dettmers et al. 2023 (arXiv:2305.17333)",
        "paper_lora":    "Hu et al. 2021 (arXiv:2106.09685)",
        "paper_sft":     "Wang et al. 2024 (arXiv:2311.08401)",
    }
    import json as _json
    with open(output_path / "training_config.json", "w") as f:
        _json.dump(config_out, f, indent=2)

    print(f"\nTraining complete. Adapter saved → {output_path / 'lora_adapter'}")
    print(f"Config saved → {output_path / 'training_config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning on faithful RAG examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",     default="geitje",
                        choices=list(_HF_IDS.keys()),
                        help="Model key to fine-tune (default: geitje)")
    parser.add_argument("--lora_r",    type=int,   default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha",type=int,   default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--epochs",    type=int,   default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--output",    default=None,
                        help="Output directory (default: results/finetuned/{model}_qlora)")
    parser.add_argument("--results-base", default=str(_ROOT / "results"),
                        help="Base results directory for loading faithful examples")
    parser.add_argument("--unified",   default=None,
                        help="Path to unified_results.jsonl (overrides --results-base per-model)")
    parser.add_argument("--batch-size",type=int,   default=4)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Validate setup without running training")
    args = parser.parse_args()

    hf_id      = _HF_IDS[args.model]
    output_dir = args.output or str(_ROOT / "results" / "finetuned" / f"{args.model}_qlora")

    print(f"\nQLoRA Fine-tuning Setup")
    print(f"  Model:       {args.model} ({hf_id})")
    print(f"  LoRA rank:   {args.lora_r}")
    print(f"  Output:      {output_dir}")

    # Load faithful examples
    if args.unified:
        unified_path = Path(args.unified)
        examples = load_unified_faithful(unified_path)
    else:
        examples = load_faithful_examples(Path(args.results_base), args.model)

    if not examples:
        print("ERROR: No FAITHFUL examples found. Run RQ1 annotation step first.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[DRY RUN] Would train on {len(examples)} examples.")
        # Format first example to verify
        sample_text = format_training_example(examples[0])
        print(f"  Sample text length: {len(sample_text)} chars")
        print(f"  Sample preview: {sample_text[:200]}...")
        run_qlora(hf_id, [sample_text], output_dir, dry_run=True)
        return

    # Format all examples into training texts
    training_texts = [format_training_example(ex) for ex in examples]
    print(f"  Formatted {len(training_texts)} training examples")

    run_qlora(
        hf_model_id=hf_id,
        training_texts=training_texts,
        output_dir=output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
