# Data Layout

This tree separates local raw material from transformed training data.

- `raw/manifests/`: provenance manifests that describe where raw records came from and whether they are safe to transform.
- `raw/scraped/`: local scrape cache for third-party pages. Keep payloads local and out of Git.
- `raw/self_authored/`: owned or synthetic examples that can be safely committed.
- `interim/`: cleaned but not yet final records.
- `processed/`: training-ready JSONL emitted by `scripts/build_dataset.py`.
- `eval/`: held-out prompts and scoring fixtures.

The canonical flow is:

1. put owned or permitted raw records under `raw/`,
2. describe them with one or more manifests under `raw/manifests/`,
3. run `python scripts/build_dataset.py --manifest-dir data/raw/manifests --output-dir data/processed`,
4. train only on `processed/` outputs after provenance review.
