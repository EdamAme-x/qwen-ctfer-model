# Processed Data

`scripts/build_dataset.py` writes training-ready JSONL files here by split.

Typical outputs:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `dataset.jsonl` when no split metadata is present

Generated outputs do not need to be committed unless they are tiny synthetic fixtures for tooling tests.
