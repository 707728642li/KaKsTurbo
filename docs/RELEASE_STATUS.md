# Release status

The public [GitHub pre-release](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) provides **Windows 1.1.0rc10** and **Linux 1.1.0rc9-tutorial1**, their corresponding source archives, a Windows extraction helper and SHA256SUMS. [PyPI 1.1.0rc10](https://pypi.org/project/kaksturbo/1.1.0rc10/) now provides qualified Linux/Windows wheels and a source archive. [Bioconda PR #70404](https://github.com/bioconda/bioconda-recipes/pull/70404) has passed upstream lint, Linux native builds and three-pair recipe tests on Python 3.11/3.12/3.13; channel availability still depends on upstream review, merge and upload. The portable platform versions remain intentionally distinct; both PyPI wheels use rc10.

## Qualification retained

Windows rc10 covers spaces/Chinese characters, 389-character input paths, 374-character output directories, 519-character result files, direct EXE launch from a 199-character installation and PowerShell 5.1 launch from a 334-character installation. LongPathsEnabled remained zero. CPU/GPU/reproduction, metadata, gzip, compatible export and actual interrupted recovery were checked; 58 interface tests passed. Nine standard scientific files were byte-identical to the corresponding rc9 backend/mode.

Linux retains its glibc 2.17 ABI target, native Ubuntu 22.04 CPU and Ubuntu 22.04 GPU-container qualification, including 1,000-pair and 49-pair backend-specific regressions. Four hard-panel model fits remain above the strict projected-gradient threshold. The raw-CDS tutorial passed the full alignment-to-fitting workflow on four synthetic requests: three finite estimates and one NA. [Platform details](LINUX_COMPATIBILITY.md) · [Tutorial qualification](RAW_CDS_VALIDATION.md)

These checks qualify installation and compatibility; they are not new performance measurements or global-convergence claims.

## Public documentation update

The repository homepage now includes the explicitly authorized workflow and performance figures, traceable aggregate timing tables, methods documentation, mode selection and a real synthetic-result preview. Original figure bytes, historical measurements and scientific source are preserved. Current `docs2` portable archives include the updated documentation, figures, aggregate data and synthetic preview. Their executables, runtime code, helper scripts and scientific settings are byte-identical to the original portable archives. All previously published assets, including the original and docs1 portable archives, and the release tag are preserved. The versioned source archives remain their original snapshots; current documentation also lives on `main`.

Use `SHA256SUMS-docs2.txt` for the new archives, and retain the original `SHA256SUMS.txt` for the original downloads. The docs1 archives retain their own checksum and qualification files. Each new archive includes `DOCUMENTATION_UPDATE.json` and a complete `FILES.sha256.json`; historical qualification records remain unchanged. New extraction, local-link and actual small CPU workflow checks are reported separately in `KaKsTurbo-docs2-QUALIFICATION.json` on the Release. These checks rerun no performance benchmark or GPU fit.

The repository includes no private research sequences, manuscript documents or large benchmark intermediates. Only the two requested scientific figures and their aggregate performance evidence have been added to the public software presentation.
