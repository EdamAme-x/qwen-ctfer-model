# qwen-ctfer-model

Qwen を LoRA/QLoRA で CTF 向けに調整し、ローカル運用と Hugging Face 公開までを再現可能にするためのリポジトリです。

## Current stance

- 実装ベースライン: `Qwen/Qwen2.5-Coder-7B-Instruct`
- 比較昇格候補: `Qwen/Qwen3.5-9B`
- 学習方式: supervised fine-tuning with LoRA/QLoRA
- 公開優先度: adapter first, merged model second

パイプラインはまず `Qwen2.5-Coder-7B-Instruct` で固め、同じデータと同じスクリプトで `Qwen3.5-9B` を比較できるようにします。

## Planned layout

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── eval/
│   └── train/
├── data/
│   ├── eval/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       ├── manifests/
│       ├── scraped/
│       └── self_authored/
├── scripts/
│   ├── build_dataset.py
│   ├── create_worktrees.sh
│   ├── merge_adapter.py
│   ├── push_to_hub.py
│   ├── run_eval.py
│   └── train_lora.py
└── outputs/
```

## Data policy

- `data/raw/self_authored/` には自作 writeup、exploit transcript、solver logs を置きます。
- `data/raw/scraped/` は自動取得した writeup のローカルキャッシュです。Git には載せません。
- `data/raw/manifests/` には raw source ごとの provenance manifest を置きます。
- 学習に入れるのは最終的に `data/processed/` の JSONL のみです。

第三者 writeup の原文をそのまま公開 dataset に含める前提ではありません。

## Quickstart

### 1. Python 環境

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
bash scripts/install_hooks.sh
```

### 2. 5 worker 用 worktree を切る

```bash
bash scripts/create_worktrees.sh
```

既定ではリポジトリの親ディレクトリに `qwen-ctfer-model-worktrees/` を作り、以下の 5 branch/worktree を生成します。

- `feat/dataset`
- `feat/train`
- `feat/eval`
- `feat/release`
- `feat/docs`

別の親ディレクトリや基点 branch を使う場合:

```bash
bash scripts/create_worktrees.sh --base-dir ../wt --base-ref main
```

### 3. Raw データを配置する

```text
data/raw/self_authored/
data/raw/scraped/
data/raw/manifests/
```

最低でも manifest を用意し、`source_url`、`fetched_at`、`allowed_for_training`、`allowed_for_redistribution` を追えるようにします。
大会名や問題名を学習で覚え込ませたくない場合は、強匿名化を前提にします。

### 4. 学習用 dataset を作る

```bash
python scripts/build_dataset.py \
  --manifest-dir data/raw/manifests \
  --output-dir data/processed \
  --strict-anonymize \
  --redaction-rules data/raw/manifests/redaction_rules.example.json
```

想定出力:

- `data/interim/*.jsonl`
- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`

`--strict-anonymize` は provenance と title 系 metadata を落とします。`--redaction-rules` は本文中の大会名や問題名を regex で `<CTF_EVENT>` や `<CHALLENGE_NAME>` に置き換えるためのフックです。

### 5. LoRA/QLoRA 学習

```bash
python scripts/train_lora.py --config configs/train/qwen25-coder-7b-qlora.json
```

`model_id` を差し替えるだけで `Qwen/Qwen3.5-9B` 比較に進める想定です。

### 6. 評価

```bash
python scripts/run_eval.py --config configs/eval/qwen25_coder_7b_base.json
```

評価では最低でも次を見ます。

- exact/contains/regex ベースの自動判定
- base vs tuned の比較
- 手動レビュー用 JSON レポート

### 7. マージと Hugging Face 公開

```bash
python scripts/merge_adapter.py --help
python scripts/push_to_hub.py --help
```

adapter を先に公開し、merged model は後追いにします。

### 8. Hugging Face token を `.env` で渡す

`.env.example` を参考にリポジトリ直下へ `.env` を置きます。

```bash
cp .env.example .env
```

最低限:

```dotenv
HUGGING_FACE_TOKEN=hf_xxx
```

公開スクリプトは `HUGGING_FACE_TOKEN` を優先して使い、互換のため `HF_TOKEN` と `HUGGING_FACE_HUB_TOKEN` も受け付ける前提です。

### 9. Model card を埋める

公開前に `model_cards/adapter.md` または `model_cards/merged.md` をベースにして、実際に upload するフォルダの `README.md` を作ります。

埋める内容:

- base model と license
- 学習データの由来と匿名化方針
- 主な学習設定
- eval 結果
- 制限事項と intended use

### 10. Adapter を公開する

adapter release directory の例:

```text
outputs/checkpoints/qwen25-coder-7b-qlora/
├── README.md
├── adapter_config.json
└── ...
```

公開例:

```bash
python scripts/push_to_hub.py \
  --local-dir outputs/checkpoints/qwen25-coder-7b-qlora \
  --repo-id <hf_user>/qwen-ctfer-7b-lora \
  --release-kind adapter
```

### 11. Merged model を公開する

まず adapter をマージします。

```bash
python scripts/merge_adapter.py \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter outputs/checkpoints/qwen25-coder-7b-qlora \
  --output-dir outputs/merged/qwen25-coder-7b
```

merged release directory の例:

```text
outputs/merged/qwen25-coder-7b/
├── README.md
├── config.json
└── ...
```

公開例:

```bash
python scripts/push_to_hub.py \
  --local-dir outputs/merged/qwen25-coder-7b \
  --repo-id <hf_user>/qwen-ctfer-7b-merged \
  --release-kind merged
```

## Minimal workflow

1. raw source と provenance manifest を収集する
2. `build_dataset.py` で学習用 JSONL に変換する
3. `train_lora.py` で `Qwen2.5-Coder-7B-Instruct` をベースに adapter を学習する
4. `run_eval.py` で held-out set を評価する
5. `merge_adapter.py` で必要なら merged model を作る
6. `push_to_hub.py` で adapter を公開する
7. `model_id` を差し替えて `Qwen3.5-9B` を同条件比較する

## Collaboration rule

この repo は 5 worker 分担前提です。1 変更は 1 owned slice に閉じ、どんなに小さくても commit/push を刻みます。親は統合と cross-cutting change だけを持ちます。
