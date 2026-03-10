#!/usr/bin/env python3
"""Merge a PEFT adapter into a base causal LM and save merged weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


DTYPE_CHOICES = {
    "auto": None,
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA/PEFT adapter into a base model checkpoint."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model id or local path, for example Qwen/Qwen2.5-Coder-7B-Instruct.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter path or Hub repo id produced by PEFT training.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=sorted(DTYPE_CHOICES),
        help="Torch dtype for loading the base model before merge.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map passed to Transformers, for example "auto" or "cuda:0".',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        help="Enable PEFT safe_merge checks before writing merged weights.",
    )
    parser.add_argument(
        "--save-tokenizer-from",
        choices=("base", "adapter"),
        default="base",
        help="Source to use when saving tokenizer files.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return None

    import torch

    return getattr(torch, DTYPE_CHOICES[dtype_name])


def load_tokenizer(model_ref: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch_dtype = resolve_torch_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(model, args.adapter)
    merged_model = peft_model.merge_and_unload(safe_merge=args.safe_merge)
    merged_model.save_pretrained(str(output_dir))

    tokenizer_ref = args.base_model if args.save_tokenizer_from == "base" else args.adapter
    tokenizer = load_tokenizer(tokenizer_ref, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(str(output_dir))

    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
