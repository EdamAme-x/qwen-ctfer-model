#!/usr/bin/env python3
"""Run prompt-based evaluation for a base model or a PEFT adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    type: str
    expected: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JSONL prompt evaluations against a base model or LoRA adapter."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file that defines model and evaluation settings.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                record["id"] = f"case_{index:04d}"
            cases.append(record)
    if not cases:
        raise ValueError(f"No evaluation cases were found in {path}")
    return cases


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def derive_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "eval" and config_path.parent.parent.name == "configs":
        return config_path.parent.parent.parent
    return Path.cwd()


def resolve_project_path(project_root: Path, raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def resolve_dtype(dtype_name: str | None) -> Any:
    if not dtype_name or dtype_name == "auto":
        return "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model_name_or_path"]
    trust_remote_code = bool(config.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=config.get("revision"),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "revision": config.get("revision"),
        "trust_remote_code": trust_remote_code,
        "device_map": config.get("device_map", "auto"),
        "attn_implementation": config.get("attn_implementation"),
    }
    torch_dtype = resolve_dtype(config.get("torch_dtype"))
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    adapter_path = config.get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def render_prompt(
    case: dict[str, Any],
    tokenizer: Any,
    default_system_prompt: str | None,
) -> tuple[str, bool]:
    messages = case.get("messages")
    if messages:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, True

    prompt = case.get("prompt")
    if prompt is None:
        raise ValueError(f"Case {case['id']} must define either 'messages' or 'prompt'")

    if default_system_prompt:
        messages = [
            {"role": "system", "content": default_system_prompt},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, True

    return str(prompt), False


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    generation_config: dict[str, Any],
) -> str:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    output = model.generate(
        **encoded,
        max_new_tokens=generation_config.get("max_new_tokens", 256),
        temperature=generation_config.get("temperature", 0.0),
        top_p=generation_config.get("top_p", 1.0),
        do_sample=generation_config.get("do_sample", False),
        repetition_penalty=generation_config.get("repetition_penalty", 1.0),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_tokens = encoded["input_ids"].shape[-1]
    completion_tokens = output[0][prompt_tokens:]
    return tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()


def normalize_checks(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "checks" in case:
        checks = case["checks"]
    else:
        checks = []
        for key in ("exact", "contains", "regex"):
            if key in case:
                checks.append({"type": key, "value": case[key]})
    if not checks:
        raise ValueError(f"Case {case['id']} has no checks")
    return checks


def evaluate_check(text: str, check: dict[str, Any]) -> CheckResult:
    check_type = check["type"]
    expected = str(check["value"])

    if check_type == "exact":
        passed = text.strip() == expected.strip()
        detail = "exact match"
    elif check_type == "contains":
        passed = expected in text
        detail = "substring match"
    elif check_type == "regex":
        flags = 0
        for flag_name in check.get("flags", []):
            flags |= getattr(re, flag_name)
        passed = re.search(expected, text, flags=flags) is not None
        detail = "regex match"
    else:
        raise ValueError(f"Unsupported check type: {check_type}")

    return CheckResult(
        type=check_type,
        expected=expected,
        passed=passed,
        detail=detail,
    )


def evaluate_case(
    case: dict[str, Any],
    output_text: str,
) -> dict[str, Any]:
    check_results = [evaluate_check(output_text, check) for check in normalize_checks(case)]
    passed = all(item.passed for item in check_results)
    return {
        "id": case["id"],
        "category": case.get("category"),
        "passed": passed,
        "checks": [
            {
                "type": item.type,
                "expected": item.expected,
                "passed": item.passed,
                "detail": item.detail,
            }
            for item in check_results
        ],
        "metadata": case.get("metadata", {}),
        "response": output_text,
    }


def summarize_results(
    case_results: list[dict[str, Any]],
    started_at: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for item in case_results if item["passed"])
    failed = total - passed
    by_category: dict[str, dict[str, int]] = {}
    for item in case_results:
        category = item.get("category") or "uncategorized"
        stats = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        stats["total"] += 1
        if item["passed"]:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

    finished_at = time.time()
    return {
        "model_name_or_path": config["model_name_or_path"],
        "adapter_path": config.get("adapter_path"),
        "cases_path": config["cases_path"],
        "report_path": config["report_path"],
        "duration_seconds": round(finished_at - started_at, 3),
        "totals": {
            "cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        "by_category": by_category,
        "generation": config.get("generation", {}),
        "results": case_results,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    project_root = derive_project_root(config_path)

    config["cases_path"] = resolve_project_path(project_root, config["cases_path"])
    config["report_path"] = resolve_project_path(project_root, config["report_path"])
    config["adapter_path"] = resolve_project_path(project_root, config.get("adapter_path"))

    cases_path = Path(config["cases_path"]).resolve()
    report_path = Path(config["report_path"]).resolve()
    ensure_parent(report_path)

    model, tokenizer = load_model_and_tokenizer(config)
    cases = load_cases(cases_path)
    generation_config = config.get("generation", {})
    default_system_prompt = config.get("default_system_prompt")

    started_at = time.time()
    case_results: list[dict[str, Any]] = []
    for case in cases:
        prompt_text, _ = render_prompt(case, tokenizer, default_system_prompt)
        output_text = generate_one(model, tokenizer, prompt_text, generation_config)
        case_results.append(evaluate_case(case, output_text))

    report = summarize_results(case_results, started_at, config)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(report["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
