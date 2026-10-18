# Release status

The public [GitHub pre-release](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) provides **Windows 1.1.0rc10** and **Linux 1.1.0rc9-tutorial1**, their corresponding source archives, a Windows extraction helper and SHA256SUMS. PyPI and Bioconda publication remain pending. The platform program versions are intentionally distinct.

## Qualification retained

Windows rc10 covers spaces/Chinese characters, 389-character input paths, 374-character output directories, 519-character result files, direct EXE launch from a 199-character installation and PowerShell 5.1 launch from a 334-character installation. LongPathsEnabled remained zero. CPU/GPU/reproduction, metadata, gzip, compatible export and actual interrupted recovery were checked; 58 interface tests passed. Nine standard scientific files were byte-identical to the corresponding rc9 backend/mode.

Linux retains its glibc 2.17 ABI target, native Ubuntu 22.04 CPU and Ubuntu 22.04 GPU-container qualification, including 1,000-pair and 49-pair backend-specific regressions. Four hard-panel model fits remain above the strict projected-gradient threshold. The raw-CDS tutorial passed the full alignment-to-fitting workflow on four synthetic requests: three finite estimates and one NA. [Platform details](LINUX_COMPATIBILITY.md) · [Tutorial qualification](RAW_CDS_VALIDATION.md)

These checks qualify installation and compatibility; they are not new performance measurements or global-convergence claims.

## Public documentation update

The repository homepage now includes the explicitly authorized workflow and performance figures, traceable aggregate timing tables, methods documentation, mode selection and a real synthetic-result preview. Original figure bytes, historical measurements and scientific source are preserved. New `docs1` portable archives include the updated documentation, figures, aggregate data and synthetic preview. Their executables, runtime code, helper scripts and scientific settings are byte-identical to the original portable archives. The original six assets and release tag are preserved. The versioned source archives remain their original snapshots; current documentation also lives on `main`.

Use `SHA256SUMS-docs1.txt` for the new archives, and retain the original `SHA256SUMS.txt` for the original downloads. Each new archive includes `DOCUMENTATION_UPDATE.json` and a complete `FILES.sha256.json`; historical qualification records remain unchanged. New extraction, local-link and actual small CPU workflow checks are reported separately in `KaKsTurbo-docs1-QUALIFICATION.json` on the Release. These checks rerun no performance benchmark or GPU fit.

The repository includes no private research sequences, manuscript documents or large benchmark intermediates. Only the two requested scientific figures and their aggregate performance evidence have been added to the public software presentation.
