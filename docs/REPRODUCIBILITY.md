# Reproducibility

The frozen scientific engine remains 1.1.0rc4+schur.1. Platform-specific complete asset manifests are pinned in engine_pins.json. Packaging identity 1.1.0rc8 differs from the previous local-suffix interface identity, so old checkpoints are not silently migrated. Runtime verification checks all listed file sizes/hashes.

Preserve input bytes, directed IDs/order, NA outcomes, fourteen models, FP64, fixed bounds, 500 fitting iterations and all quality gates. Package smoke tests do not replace full scientific regression. Baseline speedups belong to the archived server experiments; GPU timings belong to Windows with their recorded concurrency. No new benchmark is included in release qualification.
