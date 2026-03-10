# qwen-ctfer-model Design

## 1. Goal

This repository exists to build a CTF-focused local model by fine-tuning a Qwen base model with LoRA/QLoRA, then publishing the resulting artifact to Hugging Face in a reproducible way.

The v1 target is not "the strongest model possible". The v1 target is:

- a model that is better than the base model on CTF-style tasks,
- cheap enough to train locally,
- small enough to iterate on quickly,
- publishable with a clean model card and reproducible pipeline.

## 2. Product Definition

The intended model behavior is:

- generate exploit scaffolds, solver code, and triage plans for CTF tasks,
- reason over pwn, rev, web, crypto, forensics, and OSINT prompts,
- produce compact, operator-friendly answers instead of generic assistant prose,
- preserve enough general coding ability to remain useful as a local daily driver.

The model is not intended to be:

- a full-parameter foundation-model retrain,
- a broad offensive model for uncontrolled real-world intrusion use,
- a benchmark vanity project without task-specific evaluation.

## 3. Base Model Choice

### Implementation baseline

Use `Qwen/Qwen2.5-Coder-7B-Instruct` as the implementation baseline.

Why this is the first implementation target:

- Apache-2.0 license is simple for downstream distribution.
- The official model card states the model is code-focused, retains math and general capabilities, and supports long context.
- The official Qwen docs support LoRA and QLoRA training flows for Qwen models.
- 7B remains realistic for local adapter tuning and local inference.
- For a CTF assistant, code generation and code repair matter enough to justify preferring the coder-tuned 7B line first.

### Promotion candidate

After the end-to-end pipeline works on the baseline, the first promoted comparison target should be `Qwen/Qwen3.5-9B`.

Why this is the next comparison target:

- it satisfies the repository size floor,
- it is a stronger general candidate for reasoning and agent-style behavior,
- it can be evaluated with the same SFT, eval, merge, and Hub release pipeline,
- it gives a clean apples-to-apples comparison once the data and scripts are already stable.

The project should not start with `Qwen3.5-9B` before the pipeline is proven on `Qwen2.5-Coder-7B-Instruct`. The goal is to de-risk implementation first, then compare model quality.

### Secondary alternative base model

If the project later decides that general reasoning and agent behavior matter more than code-specialized behavior, the main alternative is `Qwen/Qwen3-8B`.

That alternative should be treated as a deliberate design revision, not as a casual swap, because eval results will no longer be directly comparable.

### Size floor

Do not use sub-7B models in this repository.

The minimum acceptable model size for this project is:

- `7B` class for Qwen2.5
- `8B` class for Qwen3

The repository should optimize for quality and local usability at that floor, not for laptop-class minimum-VRAM experiments.

## 4. Training Strategy

### v1 recipe

Use supervised fine-tuning first. Do not start with RLHF, DPO, or GRPO.

The first release should be:

1. base model,
2. curated CTF conversational dataset,
3. QLoRA or LoRA adapter training,
4. held-out evaluation,
5. adapter release to Hugging Face,
6. optional merged model release for local inference.

### Why LoRA/QLoRA first

- cheaper than full fine-tuning,
- easier to reproduce,
- easier to compare against the base model,
- easier to publish as an adapter,
- lower risk of destroying the base model's general competence.

### Training stack

Primary stack:

- `transformers`
- `trl`
- `peft`
- `datasets`
- `accelerate`
- `bitsandbytes`

Optional acceleration path:

- `unsloth` for faster low-VRAM iteration

Do not make LLaMA-Factory or Unsloth the only supported path in v1. The canonical pipeline should remain plain Hugging Face libraries so the repo is easier to maintain and customize.

## 5. Planned Training Stages

### Stage A: dataset curation

Build a CTF-oriented SFT corpus with high signal and consistent answer style.

### Stage B: adapter training

Train a LoRA/QLoRA adapter on the curated set.

### Stage C: evaluation

Compare base vs adapter on held-out challenge families.

### Stage D: release

Publish at least the adapter repo. Publish the merged repo only if storage and validation are acceptable.

### Stage E: optional post-training

Only after Stage C is solid:

- preference tuning with judged pairs,
- reward-based tuning for exploit/code correctness,
- task-specific continued training for a single family such as pwn or web.

## 6. Dataset Design

### Dataset principles

The dataset matters more than hyperparameter micro-tuning.

The dataset should bias toward:

- exploit planning,
- solver construction,
- debugging failed approaches,
- binary and web triage,
- concise writeup-style explanation,
- tool-aware operational reasoning.

The dataset should avoid:

- raw scraped text dumps with unclear license,
- public flags or secrets that the model can memorize,
- contest names and challenge titles copied verbatim when they are not necessary for the skill being trained,
- low-signal generic coding chat,
- repetitive paraphrases of the same challenge,
- unverifiable or hallucinated solutions.

### Preferred sample types

Include a mix of:

- short tactical Q&A,
- long-form multi-step reasoning,
- exploit scaffold generation,
- patch or diff explanation,
- debugging traces,
- "failed attempt -> diagnosis -> corrected attempt" samples,
- tool-use planning prompts that describe what to inspect next.

### Data source priority

Preferred sources, in order:

1. self-authored writeups and solve notes,
2. self-authored exploit scripts and debugging transcripts,
3. permissively licensed public CTF data,
4. transformed derivative examples created from allowed source material.

Do not mirror arbitrary third-party writeups into a public dataset unless the license clearly allows redistribution.

### Repository data layout

Use the following storage policy:

- `data/raw/self_authored/`: local notes, solve logs, exploit transcripts, and owned material
- `data/raw/scraped/`: automated scrape cache for writeups and other reference pages
- `data/raw/manifests/`: provenance manifests for every raw source batch
- `data/interim/`: cleaned intermediate records before final conversation shaping
- `data/processed/`: final JSONL datasets used for training and evaluation

Rules for `data/raw/scraped/`:

- treat it as a local cache, not as publishable training data,
- do not commit third-party scrape payloads to Git,
- attach a manifest with URL, fetch time, author if known, and redistribution status,
- train only on transformed or self-authored derivatives after license review.
- default to strong anonymization before writing `data/processed/`: drop provenance/title-like metadata and regex-mask competition or challenge identifiers when they are not essential.

### Data schema

The canonical training format should be JSONL with one sample per line and a conversational `messages` field.

Recommended schema:

```json
{
  "id": "pwn_stack_0001",
  "category": "pwn",
  "difficulty": "medium",
  "source": "self_authored",
  "license": "internal_or_permitted",
  "challenge_family": "stack",
  "messages": [
    {
      "role": "system",
      "content": "You are a CTF assistant working in an authorized sandbox. Be concise, evidence-first, and technical."
    },
    {
      "role": "user",
      "content": "Given this ELF behavior and crash trace, find the shortest exploitation path."
    },
    {
      "role": "assistant",
      "content": "First confirm NX/PIE/canary, then recover the exact overwrite offset..."
    }
  ],
  "artifacts": {
    "language": "python",
    "has_code": true
  }
}
```

### Split policy

Never split randomly at the row level when multiple rows come from the same challenge.

Split by challenge or source bundle so train and eval do not share:

- the same flag,
- the same binary,
- the same exploit path,
- the same writeup rewritten in different wording.

## 7. Conversation Format Rules

Use Qwen's existing chat template first. Do not invent a custom template in v1 unless a concrete need appears.

Default format:

- structured conversational `messages`,
- assistant answers only in the target style,
- system prompts that explicitly frame the task as CTF or sandbox analysis.

Loss policy:

- prefer assistant-only loss when the chat template behavior is validated,
- otherwise fall back to prompt-completion style so only assistant completions are trained.

Keep the tokenizer and EOS handling aligned with Qwen conventions during training and inference.

## 8. Hyperparameter Baseline

These are starting points, not fixed law.

### QLoRA baseline

- `load_in_4bit=True`
- `bf16=True` if hardware supports it
- `gradient_checkpointing=True`
- `target_modules="all-linear"`
- `lora_r=64`
- `lora_alpha=128`
- `lora_dropout=0.05`
- context length: start at `4096`
- epochs: `1` to `3`
- learning rate: start near `2e-4` for adapters
- warmup ratio: `0.03`
- weight decay: `0.0` or very small
- packing: enable only after confirming it does not hurt answer formatting

### Batch strategy

Keep per-device batch size small and scale with gradient accumulation. Optimize for stable training and reproducibility before chasing throughput.

### Context policy

Do not jump to 32k or 128k training in v1. Most CTF supervision examples are better served by 4k to 8k contexts, and the saved memory should be spent on more clean examples and more eval cycles.

## 9. Evaluation Plan

Evaluation must decide whether the adapter is genuinely better for CTF work, not just whether loss goes down.

### Held-out task buckets

Maintain a held-out set across at least:

- pwn,
- reverse engineering,
- web exploitation,
- crypto,
- forensics or OSINT.

### What to score

Score each response for:

- technical correctness,
- shortest-path usefulness,
- code or payload executability,
- hallucination rate,
- unnecessary verbosity,
- ability to state uncertainty when evidence is missing.

### Preferred eval methods

Use a combination of:

- exact or regex checks for structured answers,
- unit tests for generated code when possible,
- rubric grading for exploit plans,
- pairwise comparison against the base model,
- a small manual review set for "would I actually use this in a CTF round?"

### Release gate

Do not publish a release candidate unless it:

- clearly beats the base model on the held-out CTF rubric,
- does not materially regress on basic coding prompts,
- passes a manual sanity pass on at least 20 to 50 representative prompts.

## 10. Repo Layout

The repo should evolve toward this structure:

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── train/
│   └── eval/
├── data/
│   ├── raw/
│   │   ├── manifests/
│   │   ├── scraped/
│   │   └── self_authored/
│   ├── interim/
│   ├── processed/
│   └── eval/
├── scripts/
│   ├── build_dataset.py
│   ├── train_lora.py
│   ├── merge_adapter.py
│   ├── run_eval.py
│   └── push_to_hub.py
├── outputs/
│   ├── checkpoints/
│   ├── merged/
│   └── reports/
└── model_cards/
    ├── adapter.md
    └── merged.md
```

Large checkpoints and temporary outputs should never be committed to Git. Keep the repository code-first, not artifact-first.

## 11. Collaboration Workflow

This repository should support five parallel workers with disjoint ownership.

### Default split

Use one branch or worktree per feature slice:

1. dataset ingestion and provenance
2. training pipeline
3. evaluation pipeline
4. merge and Hugging Face release
5. documentation and operator tooling

### Branch and worktree policy

Preferred layout:

- parent branch: integration branch, usually `main`
- worker branches: `feat/dataset`, `feat/train`, `feat/eval`, `feat/release`, `feat/docs`
- worker worktrees: sibling directories outside the repo root

If a task is small enough to stay in one branch, still keep ownership separated by file set.

### Commit policy

Every feature change, however small, should be:

1. implemented in one owned slice,
2. committed immediately with a narrow message,
3. pushed immediately after local verification,
4. merged only after the parent confirms the write set is still isolated.

Do not batch unrelated fixes into one commit.

### Integration policy

The parent agent owns:

- `AGENTS.md`
- cross-cutting refactors
- conflict resolution
- final integration and release notes

Workers should avoid editing shared files unless explicitly assigned.

## 12. Hugging Face Release Plan

### Repositories to publish

At minimum publish:

- one model repo for the LoRA adapter

Optionally publish:

- one model repo for merged weights
- one dataset repo for redistributable processed data
- one eval repo or report artifact if the results are worth sharing

### Recommended naming

- adapter: `<hf_user>/qwen-ctfer-7b-lora`
- merged: `<hf_user>/qwen-ctfer-7b-merged`
- dataset: `<hf_user>/qwen-ctfer-sft-data`

### What the adapter repo must contain

- adapter weights
- adapter config
- base model reference
- exact training stack versions
- training command or config
- evaluation summary
- intended use and limitations
- safety and license notes

### What the merged repo must contain

- merged model weights
- tokenizer files if changed
- generation example
- hardware notes for local inference
- explicit statement that it was merged from the released adapter and base model

### Upload method

Prefer scripted uploads so release is reproducible.

The release script should:

1. create or validate the Hub repo,
2. upload a local folder,
3. exclude logs, optimizer states, and junk checkpoints,
4. write a deterministic commit message,
5. fail loudly on missing metadata.

## 13. Model Card Requirements

Every public release must document:

- base model and license,
- fine-tuning method,
- dataset source categories,
- what was included and excluded,
- hardware used,
- main hyperparameters,
- evaluation methodology,
- known failure modes,
- intended use: CTF, sandbox, lab, education,
- non-goals and abuse caveats.

The model card should explicitly say that the model may hallucinate exploitability and must be validated against the actual target environment.

## 14. Security and Compliance

Before any public upload:

- strip API keys, tokens, cookies, and credentials,
- strip private infrastructure details,
- remove final flags from public training data unless intentionally synthetic,
- verify redistribution rights for any dataset content,
- keep a source ledger for every training shard,
- avoid publishing copyrighted writeup text unless permission is explicit.

If in doubt, publish the adapter and model card first, and keep the dataset private until provenance is clean.

## 15. Local Inference Plan

The model should be usable locally in two modes:

- correctness-first mode: merged or adapter-loaded model via `transformers`
- speed-first mode: quantized artifact later for `vLLM`, `llama.cpp`, or another local runner

Quantization is a deployment concern, not the first training milestone. Do not block the first release on GGUF/AWQ/GPTQ work.

## 16. Milestones

### Milestone 1: foundation

- choose final base model
- lock dependency versions
- define dataset schema
- implement raw-to-processed dataset pipeline

### Milestone 2: first training loop

- train one adapter baseline
- save training config and report
- run base vs tuned comparison on held-out prompts

### Milestone 3: publishable candidate

- improve dataset quality
- rerun training
- produce model card
- push adapter to Hugging Face

### Milestone 4: local usability

- merge adapter
- test local inference ergonomics
- optionally publish merged repo

## 17. Definition of Done for v1

v1 is done when all of the following are true:

- there is a reproducible training script or config,
- there is a curated and split CTF SFT dataset,
- there is a held-out evaluation report,
- there is a public or ready-to-publish Hugging Face adapter repo,
- the model is measurably more useful than the base model for CTF workflows,
- the release can be recreated from repo code and documented inputs.

## 18. Immediate Next Implementation Order

Implement in this order:

1. dataset schema and provenance ledger,
2. scrape cache and manifest handling under `data/raw/scraped`,
3. dataset builder script,
4. baseline training script with PEFT/TRL on `Qwen/Qwen2.5-Coder-7B-Instruct`,
5. evaluation harness,
6. merge script,
7. Hugging Face upload script,
8. swap only `model_id`-level assumptions to compare `Qwen/Qwen3.5-9B`,
9. README quickstart for training and inference.

## 19. Current Decisions

As of the current design, the project should assume:

- implementation baseline: `Qwen/Qwen2.5-Coder-7B-Instruct`
- first promoted comparison target: `Qwen/Qwen3.5-9B`
- minimum size floor: `7B` or larger
- training objective: SFT
- adapter method: QLoRA first, LoRA as fallback
- canonical framework: Hugging Face native stack
- public artifact priority: adapter first, merged model second
- primary success metric: better CTF task usefulness than the base model

Any later change to the base model or training stack should be treated as a deliberate design revision and documented in this file.

## 20. Implementation Notes

The following points were verified during the first end-to-end smoke run and should be treated as operational guidance, not theory:

- Template manifests under `data/raw/manifests/` must be explicitly skippable. Use `"enabled": false` on non-live template files so `build_dataset.py` does not try to ingest placeholder paths.
- `scripts/train_lora.py` should import dependencies with `importlib.import_module()`. Using `__import__(..., fromlist=["*"])` can trigger false failures with modern `transformers` builds because optional modules such as `torchaudio` may be touched during import.
- The current training pipeline pre-renders conversations into a plain `text` field before handing data to `trl.SFTTrainer`. In this shape, `assistant_only_loss=True` is invalid and must remain disabled until the trainer consumes conversational examples directly.
- Strong anonymization changes sample IDs enough to create collisions if IDs are derived from challenge names. Anonymous IDs should therefore be hash-derived rather than title-derived.
- A local smoke train for `Qwen/Qwen2.5-Coder-7B-Instruct` with QLoRA did complete on an RTX 5080 16 GB class GPU using the smoke config (`r=8`, `max_seq_length=1024`, `max_steps=1`) and produced adapter artifacts under `outputs/smoke/checkpoints/qwen25-coder-7b-smoke/`.
- The expensive part of the first smoke run is model download and initial weight loading, not the single optimizer step. Hub authentication improves that startup path materially, so authenticated downloads should be preferred before larger runs.
- WSL Python and Windows Python must be treated as separate runtime environments. Package installation in one does not provision the other. Heavy runs should stay inside one chosen runtime for the duration of the experiment.
