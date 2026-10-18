# Reproducibility

## Software identity

Use an explicit program version and archive checksum. The current public programs are Windows 1.1.0rc10 and Linux 1.1.0rc9-tutorial1. Their platform-native assets are independently pinned in [engine_pins.json](../engine_pins.json); runtime verification checks file sizes and SHA256 hashes. The package manifests retain original engine identities, including historical names.

The GitHub release tag and its downloadable assets are preserved. Later `main`-branch documentation changes do not silently replace binaries or release checksums. Record the commit as well when using source rather than a portable package.

## Scientific run identity

Preserve input bytes, directed IDs/order, metadata, NA outcomes, all fourteen model slots, FP64, bounds and the original stopping/quality policies. Save the run manifest, core results and quality summaries. Retain chunk files needed by unfinished checkpoints. A completed chunk counter is not final aggregation/commit.

`robust` and `reproduce` are different numerical policies. Formatting a result in a KaKs_Calculator-compatible layout does not make those estimators interchangeable. [Methods](METHODS.md) · [Checkpoint compatibility](MIGRATION.md)

## Timing evidence

[Performance documentation](PERFORMANCE.md) keeps the current same-host public panel (N=3), archived same-platform plant/fly CPU cohorts (N=5), and physical-core scaling (N=5) separate. The small published TSVs retain every timing observation used in the displayed summaries. Ratios use unrounded medians; ranges retain the slowest and fastest runs. The two figures are copied byte-for-byte with their [SHA256 provenance](assets/figure_provenance.json).

## Numerical evidence

Reproducible output does not establish stationarity. Model weights, bound/cap flags, fresh projected gradients and complete model-average sensitivity must be interpreted together. Finite-coordinate checks, strict first-order diagnostics and global optimality are distinct. [Validation scope and unresolved cases](NUMERICAL_VALIDATION.md)

The repository distributes software, synthetic examples and the explicitly authorized overview/performance figures with aggregate data. Private sequence data, manuscript documents and large research intermediates are excluded.
