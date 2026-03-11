#!/usr/bin/env python3
"""Run a local chat session against the base model plus qwen-ctfer adapter."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_HF_ADAPTER = "edamamex/qwen-ctfer"
DEFAULT_LOCAL_ADAPTER = "outputs/checkpoints/qwen-ctfer"
DEFAULT_SYSTEM_PROMPT = (
    "You are a CTF assistant working in an authorized sandbox. "
    "Be concise, evidence-first, and technical."
)


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{module_name}'. Install with: {install_hint}") from exc


transformers = require_dependency(
    "transformers",
    "pip install transformers accelerate peft torch",
)
peft = require_dependency("peft", "pip install peft")
torch = require_dependency("torch", "pip install torch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt", default=None, help="Single prompt. If omitted, start interactive chat.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    return parser.parse_args()


def resolve_adapter_path(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    local_path = Path(DEFAULT_LOCAL_ADAPTER)
    if local_path.exists():
        return str(local_path.resolve())
    return DEFAULT_HF_ADAPTER


def resolve_dtype(name: str | None) -> Any:
    if not name or name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise SystemExit(f"Unsupported torch dtype: {name}") from exc


def load_model(base_model: str, adapter: str, trust_remote_code: bool, device_map: str, torch_dtype: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(torch_dtype)
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model = transformers.AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    normalize_no_split_modules_for_peft(model)
    model = peft.PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def normalize_no_split_modules_for_peft(model: Any) -> None:
    no_split_modules = getattr(model, "_no_split_modules", None)
    if isinstance(no_split_modules, set):
        model._no_split_modules = sorted(no_split_modules)


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_tokens = encoded["input_ids"].shape[-1]
    completion = output[0][prompt_tokens:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def interactive_chat(args: argparse.Namespace, model: Any, tokenizer: Any) -> None:
    print(f"base_model={args.base_model}")
    print(f"adapter={resolve_adapter_path(args.adapter)}")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nuser> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        messages = build_messages(args.system_prompt, prompt)
        reply = generate_reply(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        print(f"\nassistant> {reply}")


def main() -> int:
    args = parse_args()
    adapter = resolve_adapter_path(args.adapter)
    model, tokenizer = load_model(
        base_model=args.base_model,
        adapter=adapter,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    if args.prompt:
        messages = build_messages(args.system_prompt, args.prompt)
        reply = generate_reply(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        print(reply)
        return 0

    interactive_chat(args, model, tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
