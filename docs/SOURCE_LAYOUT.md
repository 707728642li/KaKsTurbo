# Source layout

The directories below describe the **GitHub source checkout**. Portable downloads instead provide a platform launcher, `runtime/`, documentation, helpers and examples. They need no separate source checkout to run. Version-matched corresponding-source archives are attached to the [Release](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10); source/build-record directories are not promised inside every portable bundle.

- `src/`: public Python CLI and APIs, including retained compatibility aliases.
- `corresponding-source/`: pinned native translation units and headers, platform glue and build scripts. The unused external CUDA SDK header is intentionally excluded.
- `build-records/`: native build commands and engine manifests; historical local paths are provenance, not datasets.
- `scripts/`: input-preparation helpers, compatible readers and offline package building.
- `workflow/`, `examples/`: optional alignment environment and synthetic tutorial inputs.
- `tests/`: source tests; generated test output is excluded.
- `licenses/`: original license texts and dependency notices.

The release ZIP/TAR.GZ is the supported ready-to-run installation. The checkout excludes private datasets, raw benchmark outputs, environments and manuscript documents. The explicitly authorized workflow/performance figures and small aggregate timing tables are included under docs/assets and docs/data; examples/results contains only synthetic demonstration output. Source packaging needs a version-matched pinned engine through KAKSTURBO_ENGINE_DIR; no engine is downloaded or fitted by the build backend. Recompiled engines require a new scientific qualification, never an edited manifest to bypass checks.
