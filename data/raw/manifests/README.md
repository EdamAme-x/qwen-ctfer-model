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

For stronger anonymization, pair the builder with a redaction file:

```bash
python scripts/build_dataset.py \
  --manifest-dir data/raw/manifests \
  --output-dir data/processed \
  --strict-anonymize \
  --redaction-rules data/raw/manifests/redaction_rules.example.json
```

`--strict-anonymize` drops provenance plus title-like metadata keys. `--redaction-rules` then regex-masks contest and challenge names that still appear in prompts or answers.
