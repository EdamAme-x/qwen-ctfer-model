#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_smoke_pipeline.sh [options]

Options:
  --train-config PATH   Train config to start from.
                        Default: configs/train/qwen25-coder-7b-smoke.json
  --python BIN          Python interpreter to use. Default: python3, fallback: python
  --output-root PATH    Root directory for smoke outputs.
                        Default: outputs/smoke
  --skip-train          Only build the dataset and run base eval.
  --skip-base-eval      Skip base-model eval.
  --skip-adapter-eval   Skip adapter eval even if an adapter artifact exists.
  -h, --help            Show this help.

Behavior:
  1. Builds a smoke dataset with strict anonymization under outputs/smoke/processed
  2. Creates temporary train/eval configs so repo-tracked configs are untouched
  3. Runs a short training job if the train config exists and --skip-train is not set
  4. Runs base eval and adapter eval against data/eval/example_cases.jsonl when possible
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  python_bin="python"
fi

train_config="configs/train/qwen25-coder-7b-smoke.json"
output_root="outputs/smoke"
skip_train=0
skip_base_eval=0
skip_adapter_eval=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config)
      train_config="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --skip-train)
      skip_train=1
      shift
      ;;
    --skip-base-eval)
      skip_base_eval=1
      shift
      ;;
    --skip-adapter-eval)
      skip_adapter_eval=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter not found: $python_bin" >&2
  exit 1
fi

build_script="scripts/build_dataset.py"
train_script="scripts/train_lora.py"
eval_script="scripts/run_eval.py"
eval_cases="data/eval/example_cases.jsonl"
redaction_rules="data/raw/manifests/redaction_rules.example.json"

for required in "$build_script" "$train_script" "$eval_script" "$eval_cases"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file missing: $required" >&2
    exit 1
  fi
done

processed_dir="$output_root/processed"
reports_dir="$output_root/reports"
checkpoints_dir="$output_root/checkpoints"
tmp_dir="$output_root/tmp"
mkdir -p "$processed_dir" "$reports_dir" "$checkpoints_dir" "$tmp_dir"

run_step() {
  echo
  echo "[run] $*"
  "$@"
}

build_cmd=(
  "$python_bin" "$build_script"
  --manifest-dir data/raw/manifests
  --output-dir "$processed_dir"
  --strict-anonymize
)
if [[ -f "$redaction_rules" ]]; then
  build_cmd+=(--redaction-rules "$redaction_rules")
fi
run_step "${build_cmd[@]}"

if [[ ! -f "$processed_dir/train.jsonl" ]]; then
  echo "Smoke dataset build did not produce $processed_dir/train.jsonl" >&2
  exit 1
fi

if [[ ! -f "$train_config" ]]; then
  echo "Train config not found: $train_config" >&2
  echo "Pass --train-config explicitly once the smoke config exists." >&2
  exit 1
fi

patched_train_config="$tmp_dir/$(basename "${train_config%.json}")-resolved.json"
train_output_dir="$checkpoints_dir/$(basename "${train_config%.json}")"

run_step "$python_bin" - "$train_config" "$patched_train_config" "$processed_dir/train.jsonl" "$processed_dir/eval.jsonl" "$train_output_dir" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
train_file = pathlib.Path(sys.argv[3]).resolve()
eval_file = pathlib.Path(sys.argv[4]).resolve()
output_dir = pathlib.Path(sys.argv[5]).resolve()

cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["data"]["train_file"] = str(train_file)
cfg["data"]["eval_file"] = str(eval_file) if eval_file.exists() else None
cfg["training"]["output_dir"] = str(output_dir)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print(dst)
PY

run_step "$python_bin" - "$patched_train_config" <<'PY'
import json
import pathlib
import sys

cfg = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"model={cfg['model']['name_or_path']}")
print(f"train_file={cfg['data']['train_file']}")
print(f"eval_file={cfg['data'].get('eval_file')}")
print(f"output_dir={cfg['training']['output_dir']}")
PY

if [[ "$skip_train" -eq 0 ]]; then
  run_step "$python_bin" "$train_script" --config "$patched_train_config"
else
  echo "[skip] training skipped by --skip-train"
fi

model_name="$("$python_bin" - "$patched_train_config" <<'PY'
import json
import pathlib
import sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["model"]["name_or_path"])
PY
)"

write_eval_config() {
  local eval_config_path="$1"
  local adapter_path="$2"
  local report_path="$3"
  run_step "$python_bin" - "$eval_config_path" "$model_name" "$adapter_path" "$eval_cases" "$report_path" <<'PY'
import json
import pathlib
import sys

dst = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
adapter_path = sys.argv[3]
cases_path = pathlib.Path(sys.argv[4]).resolve()
report_path = pathlib.Path(sys.argv[5]).resolve()

payload = {
    "model_name_or_path": model_name,
    "adapter_path": None if adapter_path == "-" else str(pathlib.Path(adapter_path).resolve()),
    "revision": None,
    "trust_remote_code": False,
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "attn_implementation": None,
    "default_system_prompt": "You are a CTF assistant working in an authorized sandbox. Be concise, evidence-first, and technical.",
    "cases_path": str(cases_path),
    "report_path": str(report_path),
    "generation": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
    },
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(dst)
PY
}

if [[ "$skip_base_eval" -eq 0 ]]; then
  base_eval_config="$tmp_dir/base-eval.json"
  base_report="$reports_dir/base_eval.json"
  write_eval_config "$base_eval_config" "-" "$base_report"
  run_step "$python_bin" "$eval_script" --config "$base_eval_config"
else
  echo "[skip] base eval skipped by --skip-base-eval"
fi

if [[ "$skip_adapter_eval" -eq 0 ]]; then
  if [[ -f "$train_output_dir/adapter_config.json" ]]; then
    adapter_eval_config="$tmp_dir/adapter-eval.json"
    adapter_report="$reports_dir/adapter_eval.json"
    write_eval_config "$adapter_eval_config" "$train_output_dir" "$adapter_report"
    run_step "$python_bin" "$eval_script" --config "$adapter_eval_config"
  else
    echo "[skip] adapter eval skipped because no adapter artifact was found at $train_output_dir"
  fi
else
  echo "[skip] adapter eval skipped by --skip-adapter-eval"
fi

echo
echo "Smoke pipeline finished."
echo "Processed dataset: $processed_dir"
echo "Reports: $reports_dir"
echo "Training output: $train_output_dir"
