# Model Card Templates

This directory holds fill-in-ready model card templates for releases produced by this repo.

Files:

- `adapter.md`: use for LoRA/QLoRA adapter releases
- `merged.md`: use for merged full-weight releases

Recommended workflow:

1. Copy the relevant template into the release directory as `README.md`.
2. Replace every `<...>` placeholder before upload.
3. Keep the card aligned with the exact training config, eval report, and dataset provenance used for that artifact.
4. Do not publish until the card states what data was included, what was excluded, and what safety limits still apply.

Minimum sections expected by this repo:

- base model and license
- fine-tuning method
- dataset provenance and exclusions
- hardware and key hyperparameters
- evaluation method and headline results
- intended use, non-goals, and limitations
- release contents

The published card should make it obvious that CTF exploitability claims are not guarantees and must be validated in the target environment.
