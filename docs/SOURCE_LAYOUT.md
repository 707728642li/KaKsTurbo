# Source layout

The directories below describe the **GitHub source checkout**. Portable downloads instead provide a platform launcher, `runtime/`, documentation, helpers and examples. They need no separate source checkout to run. Version-matched corresponding-source archives are attached to the [Release](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10); source/build-record directories are not promised inside every portable bundle.

- `src/`: public Python CLI and APIs, including retained compatibility aliases.
- `corresponding-source/`: pinned native translation units and headers, platform glue and build scripts. The unused external CUDA SDK header is intentionally excluded.
- `engine/`: pinned scientific source and first-party GPU PTX used by the Conda source build; old machine-specific build records remain in the earlier rc10 release.
- `scripts/`: input-preparation helpers, compatible readers and offline package building.
- `workflow/`, `examples/`: optional alignment environment and synthetic tutorial inputs.
- `tests/`: source tests; generated test output is excluded.
- `licenses/`: original license texts and dependency notices.

The release ZIP/TAR.GZ is a ready-to-run installation. The checkout excludes private datasets, raw benchmark outputs, environments and manuscript documents. The explicitly authorized workflow/performance figures and small aggregate timing tables are included under docs/assets and docs/data; examples/results contains only synthetic demonstration output. The 1.1.0 Conda build compiles the supplied Linux engine source before invoking the Python backend; the backend refuses to produce a wheel from an unbuilt engine. Recompiled engines require a new scientific qualification, never an edited manifest to bypass checks.
