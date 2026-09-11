# BTL Lab implementation

- Use the existing project registry, recipe catalog and SQLite ledger. Do not create a second current-status database in dated Markdown files.
- The root `btl` launcher is the workspace entry point. Keep the control package dependency-free and avoid importing training frameworks at module import time.
- Prime-RL is the preferred RL backend and BTL RL owns its integration. BTL Adapt is the supervised fine-tuning product; Unsloth Core is its current backend. Catalog entries do not imply an installed or validated integration beyond the explicitly recorded profile.
- BTL Measure is the independent evaluation product. Recompute per-item scores and preserve incomparable/failed reports; never treat training rewards or self-reported aggregates as release evidence.
- Keep local audit success, imported evidence, hardware validation and model-quality results distinct. Never promote a result because a hash or configuration check passes.
- Preserve artifact paths and record hashes rather than copying weights into the ledger.
- Existing research source and results must retain their contents. Canonical locations are now grouped; use `.btl-workspace.json` and the private relocation manifest, not fixed parent counts. Project edits use ledger overlays; update the discovery registry only for deliberate inventory changes.
- No automatic downloads, cloud provisioning, paid model calls or public release in the local audit path.
- Research links are coordination metadata. Linking a run never alters its result, and `research ready` must remain fail-closed when arm observations, artifact hashes or required evidence are incomplete.
- Use Python 3.11+; this machine has `/opt/homebrew/bin/python3.13`. Run `python3.13 -m pytest -q` from this directory for meaningful control-layer changes. The sibling Train and Kernels packages are included by the test configuration.
- Keep modules below 500 lines. Follow the natural-writing skill for documentation.
