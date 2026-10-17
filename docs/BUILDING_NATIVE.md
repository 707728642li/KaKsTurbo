# Native source and build provenance

`corresponding-source/` contains the pinned C/C++/CUDA/Python translation units, headers and retained build scripts. `build-records/` contains original engine manifests and compiler-command records. These are historical records, including original absolute paths and status at build time; current qualification is reported separately. They are not credentials or input sequence files.

Native rebuilding is distinct from Python packaging. Reconstruct each original engine layout from its platform source snapshots and build records, map historical project paths to the new root, and use the recorded compiler flags and dependency interfaces. Preserve FP64, no fast-math, model definitions, filters and bounds. CUDA requires the recorded toolkit/architecture; Windows BLAS and Linux OpenBLAS retain their own ABI and license requirements. A rebuilt engine obtains a new manifest and needs scientific qualification before it can replace a pinned engine. Do not change a manifest merely to bypass a failed asset check.

For an offline packaging build, supply the already-qualified engine directory via KAKSTURBO_ENGINE_DIR, then use the PEP 517 backend. No native recompilation or download occurs. Native rebuilds and package-index distribution require their own qualification; this GitHub portable release is not a PyPI or Bioconda publication.

The historical source manifest includes the hash of an external CUDA SDK header that is not redistributed. Obtain the toolkit from NVIDIA when rebuilding. Original build records preserve compiler commands and source provenance; local build-time paths are historical, not required input data. Use the portable archive for direct installation.
