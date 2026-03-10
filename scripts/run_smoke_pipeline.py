#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smoke dataset, optionally train a tiny adapter, and run evals.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train/qwen25-coder-7b-smoke.json",
        help="Train config to start from. Default: configs/train/qwen25-coder-7b-smoke.json",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        help="Python interpreter to use. Default: current interpreter, then python3/python fallback",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/smoke",
        help="Root directory for smoke outputs. Default: outputs/smoke",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only build the dataset and run base eval.",
    )
    parser.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="Skip base-model eval.",
    )
    parser.add_argument(
        "--skip-adapter-eval",
        action="store_true",
        help="Skip adapter eval even if an adapter artifact exists.",
    )
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def choose_python(explicit: str | None) -> str:
    candidates = [explicit, sys.executable, shutil.which("python3"), shutil.which("python")]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        if Path(resolved).exists() or shutil.which(resolved):
            return resolved
    raise FileNotFoundError("Python interpreter not found")


def run_step(cwd: Path, *args: str) -> None:
    print()
    print("[run]", " ".join(args))
    subprocess.run(list(args), cwd=cwd, check=True)


def write_json(file_path: Path, payload: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def inspect_python_runtime(python_bin: str) -> dict[str, Any]:
    probe = """
import importlib.util
import json
import sys

payload = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "training_deps": {},
}

for name in ("transformers", "accelerate", "bitsandbytes", "peft", "trl", "datasets", "torch"):
    payload["training_deps"][name] = importlib.util.find_spec(name) is not None

if payload["training_deps"]["torch"]:
    import torch

    payload["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
    }
else:
    payload["torch"] = None

print(json.dumps(payload))
"""
    result = subprocess.run(
        [python_bin, "-c", probe],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def print_python_runtime_summary(runtime: dict[str, Any]) -> None:
    print()
    print(f"python={runtime['python_executable']}")
    print(f"python_version={runtime['python_version']}")
    torch_info = runtime.get("torch")
    if torch_info:
        print(f"torch={torch_info['version']}")
        print(f"cuda_available={torch_info['cuda_available']}")
        print(f"cuda_device_count={torch_info['cuda_device_count']}")
        print(f"bf16_supported={torch_info['bf16_supported']}")
    else:
        print("torch=missing")


def validate_training_runtime(
    python_bin: str,
    runtime: dict[str, Any],
    train_config_payload: dict[str, Any],
) -> None:
    missing_deps = [
        name for name, present in runtime["training_deps"].items() if not present
    ]
    if missing_deps:
        joined = ", ".join(missing_deps)
        print(
            "Smoke training preflight failed: the selected Python is missing required "
            f"training packages: {joined}.",
            file=sys.stderr,
        )
        print(
            "Install them into the same interpreter that `bun run smoke` will use.",
            file=sys.stderr,
        )
        print(
            f"Suggested command: {python_bin} -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(
            "If this is the wrong environment, rerun with `--python PATH_TO_ENV_PYTHON`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    torch_info = runtime.get("torch") or {}
    needs_cuda = bool(train_config_payload["model"].get("load_in_4bit", False))
    needs_bf16 = bool(train_config_payload["training"].get("bf16", False)) or (
        str(train_config_payload["model"].get("torch_dtype", "")).lower() == "bfloat16"
    )

    if needs_cuda and not torch_info.get("cuda_available", False):
        print(
            "Smoke training preflight failed: this config expects CUDA, but the selected "
            "Python sees CPU-only torch or no GPU at all.",
            file=sys.stderr,
        )
        print(
            f"Interpreter: {runtime['python_executable']} (torch {torch_info.get('version', 'missing')})",
            file=sys.stderr,
        )
        print(
            "Use a GPU-enabled environment or point the smoke runner at the correct one with "
            "`--python`.",
            file=sys.stderr,
        )
        print(
            "Typical fix on Windows for CUDA 12.8: "
            f"{python_bin} -m pip install --force-reinstall torch --index-url "
            "https://download.pytorch.org/whl/cu128",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if needs_bf16 and not torch_info.get("bf16_supported", False):
        print(
            "Smoke training preflight failed: this config requests bf16, but the selected "
            "runtime does not report bf16-capable CUDA.",
            file=sys.stderr,
        )
        print(
            f"Interpreter: {runtime['python_executable']} (torch {torch_info.get('version', 'missing')})",
            file=sys.stderr,
        )
        print(
            "Either switch to a bf16-capable GPU runtime with `--python`, or change the train "
            "config to disable bf16 for this smoke run.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    python_bin = choose_python(args.python_bin)
    runtime = inspect_python_runtime(python_bin)

    build_script = resolve_repo_path(repo_root, "scripts/build_dataset.py")
    train_script = resolve_repo_path(repo_root, "scripts/train_lora.py")
    eval_script = resolve_repo_path(repo_root, "scripts/run_eval.py")
    eval_cases = resolve_repo_path(repo_root, "data/eval/example_cases.jsonl")
    redaction_rules = resolve_repo_path(repo_root, "data/raw/manifests/redaction_rules.example.json")
    train_config = resolve_repo_path(repo_root, args.train_config)
    output_root = resolve_repo_path(repo_root, args.output_root)

    for required_file in (build_script, train_script, eval_script, eval_cases):
        if not required_file.exists():
            print(f"Required file missing: {required_file}", file=sys.stderr)
            return 1

    processed_dir = output_root / "processed"
    reports_dir = output_root / "reports"
    checkpoints_dir = output_root / "checkpoints"
    tmp_dir = output_root / "tmp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    build_args = [
        python_bin,
        str(build_script),
        "--manifest-dir",
        str(resolve_repo_path(repo_root, "data/raw/manifests")),
        "--output-dir",
        str(processed_dir),
        "--strict-anonymize",
    ]
    if redaction_rules.exists():
        build_args.extend(["--redaction-rules", str(redaction_rules)])
    run_step(repo_root, *build_args)

    train_file = processed_dir / "train.jsonl"
    eval_file = processed_dir / "eval.jsonl"
    if not train_file.exists():
        print(f"Smoke dataset build did not produce {train_file}", file=sys.stderr)
        return 1

    if not train_config.exists():
        print(f"Train config not found: {train_config}", file=sys.stderr)
        print("Pass --train-config explicitly once the smoke config exists.", file=sys.stderr)
        return 1

    patched_train_config = tmp_dir / f"{train_config.stem}-resolved.json"
    train_output_dir = checkpoints_dir / train_config.stem

    train_config_payload = json.loads(train_config.read_text(encoding="utf-8"))
    train_config_payload["data"]["train_file"] = str(train_file)
    train_config_payload["data"]["eval_file"] = str(eval_file) if eval_file.exists() else None
    train_config_payload["training"]["output_dir"] = str(train_output_dir)
    write_json(patched_train_config, train_config_payload)

    print_python_runtime_summary(runtime)
    print()
    print(f"model={train_config_payload['model']['name_or_path']}")
    print(f"train_file={train_config_payload['data']['train_file']}")
    print(f"eval_file={train_config_payload['data']['eval_file']}")
    print(f"output_dir={train_config_payload['training']['output_dir']}")

    if not args.skip_train:
        validate_training_runtime(python_bin, runtime, train_config_payload)
        run_step(repo_root, python_bin, str(train_script), "--config", str(patched_train_config))
    else:
        print("[skip] training skipped by --skip-train")

    def write_eval_config(target_path: Path, adapter_path: str | None, report_path: Path) -> None:
        write_json(
            target_path,
            {
                "model_name_or_path": train_config_payload["model"]["name_or_path"],
                "adapter_path": adapter_path,
                "revision": None,
                "trust_remote_code": False,
                "torch_dtype": "bfloat16",
                "device_map": "auto",
                "attn_implementation": None,
                "default_system_prompt": "You are a CTF assistant working in an authorized sandbox. Be concise, evidence-first, and technical.",
                "cases_path": str(eval_cases),
                "report_path": str(report_path),
                "generation": {
                    "max_new_tokens": 256,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                },
            },
        )

    if not args.skip_base_eval:
        base_eval_config = tmp_dir / "base-eval.json"
        base_report = reports_dir / "base_eval.json"
        write_eval_config(base_eval_config, None, base_report)
        run_step(repo_root, python_bin, str(eval_script), "--config", str(base_eval_config))
    else:
        print("[skip] base eval skipped by --skip-base-eval")

    adapter_config_path = train_output_dir / "adapter_config.json"
    if not args.skip_adapter_eval:
        if adapter_config_path.exists():
            adapter_eval_config = tmp_dir / "adapter-eval.json"
            adapter_report = reports_dir / "adapter_eval.json"
            write_eval_config(adapter_eval_config, str(train_output_dir), adapter_report)
            run_step(repo_root, python_bin, str(eval_script), "--config", str(adapter_eval_config))
        else:
            print(
                f"[skip] adapter eval skipped because no adapter artifact was found at {train_output_dir}"
            )
    else:
        print("[skip] adapter eval skipped by --skip-adapter-eval")

    print()
    print("Smoke pipeline finished.")
    print(f"Processed dataset: {processed_dir}")
    print(f"Reports: {reports_dir}")
    print(f"Training output: {train_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
