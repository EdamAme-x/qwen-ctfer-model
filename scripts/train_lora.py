#!/usr/bin/env python3
"""Train a LoRA or QLoRA adapter for Qwen-based CTF SFT datasets."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{module_name}'. Install with: {install_hint}") from exc


transformers = require_dependency(
    "transformers",
    "pip install transformers accelerate bitsandbytes peft trl datasets",
)
torch = require_dependency("torch", "pip install torch")
datasets_lib = require_dependency("datasets", "pip install datasets")
peft = require_dependency("peft", "pip install peft")
trl = require_dependency("trl", "pip install trl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON training config")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path to resume from",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for training.output_dir",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def derive_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "train" and config_path.parent.parent.name == "configs":
        return config_path.parent.parent.parent
    return config_path.parent


def resolve_project_path(project_root: Path, raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def resolve_torch_dtype(name: str | None) -> Any:
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
    if name not in mapping:
        raise SystemExit(f"Unsupported torch dtype: {name}")
    return mapping[name]


def build_quantization_config(model_cfg: dict[str, Any]) -> Any | None:
    if not model_cfg.get("load_in_4bit", False):
        return None
    compute_dtype = resolve_torch_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def validate_record(example: dict[str, Any], source_name: str) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source_name} record is missing a non-empty 'messages' list")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{source_name} record has a non-dict message")
        if "role" not in message or "content" not in message:
            raise ValueError(f"{source_name} record has a malformed message entry")


def render_messages(example: dict[str, Any], tokenizer: Any, source_name: str) -> dict[str, str]:
    validate_record(example, source_name)
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_dataset_splits(data_cfg: dict[str, Any], tokenizer: Any) -> Any:
    data_files: dict[str, str] = {"train": data_cfg["train_file"]}
    if data_cfg.get("eval_file"):
        data_files["eval"] = data_cfg["eval_file"]

    dataset = datasets_lib.load_dataset(
        "json",
        data_files=data_files,
    )

    processed_splits: dict[str, Any] = {}
    for split_name, split_dataset in dataset.items():
        if len(split_dataset) == 0:
            raise SystemExit(f"{split_name} split is empty: {data_files[split_name]}")
        processed_splits[split_name] = split_dataset.map(
            lambda example, split_name=split_name: render_messages(example, tokenizer, split_name),
            remove_columns=split_dataset.column_names,
            desc=f"Rendering {split_name} chat samples",
        )

    return datasets_lib.DatasetDict(processed_splits)


def filter_kwargs(callable_obj: Any, candidate_kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    allowed = set(signature.parameters)
    return {key: value for key, value in candidate_kwargs.items() if key in allowed and value is not None}


def build_training_args(training_cfg: dict[str, Any], formatting_cfg: dict[str, Any]) -> Any:
    candidate_kwargs = dict(training_cfg)
    candidate_kwargs["dataset_text_field"] = "text"
    candidate_kwargs["max_seq_length"] = training_cfg.get("max_seq_length", 4096)
    candidate_kwargs["packing"] = training_cfg.get("packing", False)
    candidate_kwargs["assistant_only_loss"] = formatting_cfg.get("assistant_only_loss", False)
    candidate_kwargs["report_to"] = training_cfg.get("report_to", ["none"])
    return trl.SFTConfig(**filter_kwargs(trl.SFTConfig, candidate_kwargs))


def build_lora_config(lora_cfg: dict[str, Any]) -> Any:
    return peft.LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = config["model"]
    training_cfg = config["training"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=model_cfg.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_cfg.get("padding_side", "right")

    quantization_config = build_quantization_config(model_cfg)
    model_kwargs = {
        "pretrained_model_name_or_path": model_cfg["name_or_path"],
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "torch_dtype": resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
        "quantization_config": quantization_config,
        "device_map": model_cfg.get("device_map", "auto"),
        "attn_implementation": model_cfg.get("attn_implementation"),
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        **{key: value for key, value in model_kwargs.items() if value is not None}
    )

    if quantization_config is not None:
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        )
    elif training_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    model.config.use_cache = False
    return model, tokenizer


def maybe_override_output_dir(config: dict[str, Any], output_dir: str | None) -> None:
    if output_dir:
        config["training"]["output_dir"] = output_dir


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    config = load_json(config_path)
    maybe_override_output_dir(config, args.output_dir)
    project_root = derive_project_root(config_path)

    required_sections = ["model", "data", "lora", "training"]
    missing_sections = [name for name in required_sections if name not in config]
    if missing_sections:
        raise SystemExit(f"Config is missing sections: {', '.join(missing_sections)}")

    config["data"]["train_file"] = resolve_project_path(
        project_root, config["data"]["train_file"]
    )
    config["data"]["eval_file"] = resolve_project_path(
        project_root, config["data"].get("eval_file")
    )
    config["training"]["output_dir"] = resolve_project_path(
        project_root, config["training"]["output_dir"]
    )

    output_dir = Path(config["training"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(config.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset_splits(config["data"], tokenizer)
    if "eval" not in dataset:
        config["training"]["evaluation_strategy"] = "no"
        config["training"].pop("eval_steps", None)

    print(
        f"Loaded dataset: train={len(dataset['train'])}"
        + (f", eval={len(dataset['eval'])}" if "eval" in dataset else "")
    )
    print(f"Model: {config['model']['name_or_path']}")
    print(f"Output dir: {output_dir}")

    training_args = build_training_args(config["training"], config.get("formatting", {}))
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["eval"] if "eval" in dataset else None,
        "peft_config": build_lora_config(config["lora"]),
        "dataset_text_field": "text",
        "max_seq_length": config["training"].get("max_seq_length", 4096),
        "packing": config["training"].get("packing", False),
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = trl.SFTTrainer(**filter_kwargs(trl.SFTTrainer, trainer_kwargs))

    save_json(output_dir / "resolved_config.json", config)
    shutil.copy2(config_path, output_dir / "source_config.json")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter training finished: {output_dir}")


if __name__ == "__main__":
    main()
