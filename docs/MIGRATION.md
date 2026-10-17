# Version and checkpoint compatibility

KaKsTurbo is the public name. `kaksforge` remains an import compatibility alias. The historical numerical engine and hash-domain identifiers keep their original names to preserve identities and reproducibility.

The distribution version `1.1.0rc8` is an index-compatible release candidate derived from interface `1.1.0rc8+schur.1`. The numerical engines remain `1.1.0rc4+schur.1`, pinned separately for Windows and Linux. This packaging-only version change is not a new accuracy or speed claim.

Resume requires the same interface version, scientific settings and pinned assets. Finish old checkpoints using their original environment; this candidate intentionally rejects version-mismatched checkpoints. Keep final result/QC/provenance tables and needed resumable chunks. Never remove checkpoint-referenced chunks from an unfinished run.
