# <merged_repo_name>

## Summary

`<merged_repo_name>` is a merged full-weight checkpoint produced by combining:

- base model: `<base_model_id>`
- adapter: `<adapter_repo_name_or_path>`

This release is intended for local inference convenience. It should be functionally traceable to the adapter release that produced it.

## Intended Use

Use this merged checkpoint for:

- local `transformers` inference
- evaluation against held-out CTF prompts
- offline experimentation where loading adapter + base separately is inconvenient

Do not treat the merged checkpoint as:

- independent evidence of exploit correctness
- a replacement for reviewing the adapter release notes and training provenance

## Source Artifacts

- Base model id: `<base_model_id>`
- Base model revision: `<revision_or_commit>`
- Base model license: `<base_model_license>`
- Adapter source: `<adapter_repo_name_or_path>`
- Merge command or script: `<scripts/merge_adapter.py invocation>`

## Training and Provenance

The merged model inherits its training data, filtering, anonymization, and evaluation claims from the adapter release.

- Adapter release reference: `<adapter_repo_url_or_commit>`
- Processed dataset summary: `<short_summary>`
- Data exclusions: credentials, tokens, private infra details, non-redistributable raw scrape content, and unnecessary contest/problem identifiers

## Merge Environment

- Hardware: `<gpu/cpu details>`
- Dtype used during merge: `<dtype>`
- Trust remote code: `<true_or_false>`
- Tokenizer source: `<base_or_adapter>`

## Evaluation

### Method

- Reference adapter eval report: `<path_or_url>`
- Spot-check after merge: `<what was checked>`

### Results

- Base vs merged summary: `<short comparison>`
- Adapter vs merged parity check: `<same / small drift / known difference>`
- Known post-merge issues: `<if any>`

## Release Contents

- `config.json`
- model weight shards or safetensors files
- tokenizer files
- generation config if present
- `<optional eval report>`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "<merged_repo_or_local_path>"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

## Limitations

- Merging does not improve correctness by itself; it only changes packaging.
- The same exploit-hallucination and environment-mismatch risks from the adapter release still apply.
- Quantization, alternate runtimes, or different generation settings may change observed behavior.

## Safety and Compliance

- Review the adapter release and base model licenses before redistribution.
- Use is intended for CTF, sandbox, lab, and other authorized settings only.
- Validate all generated code, payloads, and operational guidance before use.

## Citation

If you use this release, cite:

- Base model: `<base_model_citation>`
- Adapter source: `<adapter_citation>`
- This merged release: `<project_citation_or_repo_url>`
