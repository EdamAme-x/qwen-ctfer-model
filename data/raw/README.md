# Raw Data Rules

`raw/` contains source material, not training-ready data.

- `manifests/` should be committed. They document provenance, licensing notes, and the mapping from source files to normalized records.
- `scraped/` is a local cache for automated scrape outputs. Third-party payloads should stay uncommitted unless redistribution is clearly allowed.
- `self_authored/` may contain owned examples, synthetic examples, or transformed derivatives that are safe to check in.

Keep manifests narrow and explicit. A manifest should tell the dataset builder what file to read, what format that file uses, and what metadata defaults to apply.
