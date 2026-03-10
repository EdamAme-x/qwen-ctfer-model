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
