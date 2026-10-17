# Release status

Current portable pre-release: Windows **1.1.0rc10** and Linux **1.1.0rc9-tutorial1**. See the repository README and GitHub Release for downloads and SHA256 values. The Linux program is retained from its qualified build; it has not been relabeled rc10.

Windows long-path qualification covers spaces/Chinese characters, 389-character input paths, 374-character output directories, 519-character result files, direct EXE launch from a 199-character installation and PowerShell 5.1 launch from a 334-character installation. LongPathsEnabled remained zero. CPU/GPU/reproduction, metadata, gzip, compatible export and actual interrupted recovery were checked; 58 interface tests passed. Nine standard scientific files were byte-identical to the corresponding rc9 backend/mode. These are small installation regressions, not new scientific or performance claims.

Linux retains its glibc2.17 ABI target, native Ubuntu22.04 CPU and Ubuntu22.04 GPU-container qualification, including 1000-pair and 49-pair backend-specific regressions. Four hard-panel models remain above the strict projected-gradient threshold. The Linux raw-CDS tutorial passed the full alignment-to-fitting workflow on four synthetic requests (three finite estimates, one NA). See LINUX_COMPATIBILITY.md and RAW_CDS_VALIDATION.md for scope.

For publication, explanatory documents and inventories were updated. Scientific binaries, PTX, Python runtime/interface code, scripts and model settings remain unchanged from the qualified portable archives; PUBLICATION_PACKAGING.json records this distinction. Prior local archives are retained. No private sequence data, manuscript or benchmark intermediates are included in this repository. PyPI/Bioconda distribution is not claimed.
