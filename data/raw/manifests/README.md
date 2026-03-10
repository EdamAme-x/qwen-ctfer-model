# Manifest Format

Each manifest is a JSON file with top-level defaults plus one or more record sources.

Minimal shape:

```json
{
  "source_id": "example_batch",
  "source": "self_authored",
  "license": "internal_or_permitted",
  "defaults": {
    "category": "rev",
    "difficulty": "easy",
    "challenge_family": "warmup",
    "split": "train"
  },
  "records": [
    {
      "path": "data/raw/self_authored/example_records.jsonl",
      "format": "jsonl_qa"
    }
  ]
}
```

Supported formats in `scripts/build_dataset.py`:

- `jsonl_qa`
- `json_qa`
- `jsonl_messages`
- `json_messages`

`*_qa` formats are converted into the canonical `messages` schema. `*_messages` formats can already carry full conversational examples.

Reviewed scraped data should flow through this sequence:

1. scrape cache under `data/raw/scraped/<batch>/`
2. shared cache under `data/raw/scraped/.cache/` to avoid refetching the same URL across batches
3. review ledger generated from `<batch>/fetch_manifest.json`
4. transformed derivative JSONL created only from approved sources
5. reviewed manifest that points at that derivative JSONL
6. `build_dataset.py`

Typical operator flow:

```bash
bun run list-ctftime-writeups -- --max-pages 25 --include-original --target-source both --output-txt outputs/ctftime_batch01.txt --output-jsonl outputs/ctftime_batch01.jsonl --overwrite
bun run scrape-writeups -- --url-file outputs/ctftime_batch01.txt --batch-name ctftime_batch01 --save-body --max-concurrency 10 --cache-dir data/raw/scraped/.cache --cache-mode reuse --overwrite
bun run prepare-scrape-review -- --fetch-manifest data/raw/scraped/ctftime_batch01/fetch_manifest.json --approve-training --review-status approved_for_training --permission-basis "explicit permission from CTFtime org and linked writeup usage approval for CTF agent development" --overwrite
```

`fetch_manifest.json` records may contain `status: "ok"` or `status: "cached"`. Both mean the payload is available for later review and derivative generation.

Raw scraped payloads stay local cache only. Do not point manifests at `data/raw/scraped/` directly. Point them at reviewed derivative JSONL under a committed location such as `data/raw/self_authored/` only after review.

For reviewed scraped manifests, make the review state explicit in defaults:

- `allowed_for_training`
- `allowed_for_redistribution`
- `review_status`
- `redistribution_status`
- `source_ledger_path`

`scripts/build_dataset.py` refuses to ingest reviewed scraped manifests when those fields are missing, still pending, or point at a missing source ledger.

For stronger anonymization, pair the builder with a redaction file:

```bash
python scripts/build_dataset.py \
  --manifest-dir data/raw/manifests \
  --output-dir data/processed \
  --strict-anonymize \
  --redaction-rules data/raw/manifests/redaction_rules.example.json
```

`--strict-anonymize` drops provenance plus title-like metadata keys. `--redaction-rules` then regex-masks contest and challenge names that still appear in prompts or answers.
