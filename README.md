# KaKsTurbo

**[Download Windows / Linux](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) · [Quick start](QUICKSTART.md) · [中文快速入门](QUICKSTART_ZH.md) · [Raw CDS/PEP tutorial](docs/RAW_CDS_TUTORIAL.md)**

KaKsTurbo estimates Ka, Ks and Ka/Ks for directed, codon-aligned sequence pairs using CPU or single-GPU execution. It retains fourteen candidate models, complete model averaging, FP64 arithmetic and explicit numerical-quality diagnostics. Standard genetic code 1 is supported.

## Download and install

This is a **release candidate**, not a final stable version. Download the complete portable archive; a source checkout is not a stand-alone installation. The portable bundles include their Python runtime and CPU/GPU backends. No system Python or CUDA Toolkit installation is required for core execution.

| Platform | Qualified program version | Download |
|---|---|---|
| Linux x86-64 | 1.1.0rc9-tutorial1 | [Linux TAR.GZ](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc9-tutorial1-linux-x86_64.tar.gz) |
| Windows x86-64 | 1.1.0rc10 | [Windows ZIP](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc10-windows-x86_64.zip) |
| Windows extraction helper | PowerShell 5.1+ | [Install-KaKsTurbo.ps1](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/Install-KaKsTurbo.ps1) |

[SHA256SUMS](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/SHA256SUMS.txt) covers the downloadable files. The Windows path fix did not change the qualified Linux executable. The two platform versions are deliberately identified separately. Publication packaging updates documentation and inventories only; native engines, public runtime code and numerical settings match the qualified archives.

### Linux (primary platform)

```bash
tar -xzf KaKsTurbo-1.1.0rc9-tutorial1-linux-x86_64.tar.gz
cd KaKsTurbo-1.1.0rc9-tutorial1
./kaksturbo --version
./kaksturbo doctor --backend cpu
./kaksturbo validate --input examples/three_pair_smoke.axt
./kaksturbo run --input examples/three_pair_smoke.axt --output results_demo --backend cpu --threads 8
```

Use a new output directory. Keep the executable and its accompanying runtime together. The Linux binary targets x86-64 and glibc >=2.17. Native Ubuntu 22.04 CPU, CentOS 7 container CPU smoke checks, and Ubuntu 22.04 GPU container userspace on an Ubuntu 24.04 NVIDIA host have been tested; this does not certify all kernels, drivers or distributions. See [Linux compatibility](docs/LINUX_COMPATIBILITY.md).

### Windows

Put the ZIP and extraction helper together, then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-KaKsTurbo.ps1 -Destination "D:\My projects\KaKsTurbo"
& "D:\My projects\KaKsTurbo\kaksturbo.exe" doctor --backend cpu
& "D:\My projects\KaKsTurbo\kaksturbo.exe" run --input "D:\My projects\KaKsTurbo\examples\three_pair_smoke.axt" --output "D:\My results\run 1" --backend cpu --threads 8
```

Spaces, Chinese characters and long input/output paths are handled internally. For an installation beyond legacy shell launch limits, the helper prints the bundled PowerShell entry command. No registry edit or directory relocation is required. Read [Windows paths](docs/WINDOWS_PATHS.md) / [中文说明](docs/WINDOWS_PATHS_ZH.md) for tested limits.

## CPU and GPU selection

CPU use needs no NVIDIA hardware or CUDA. `--backend auto` currently selects CPU. For an explicit GPU run:

```bash
nvidia-smi
./kaksturbo doctor --backend gpu --probe-gpu
./kaksturbo run --input prepared --output results_gpu --backend gpu --device cuda:0
```

The precompiled GPU backend requires an NVIDIA device with compute capability >=8.6, driver CUDA compatibility >=12.9, and at least 8 GiB free GPU memory including its reserve. A separate CUDA Toolkit is not required. Explicit GPU requests return a diagnostic error when the device, driver, backend or memory is unsuitable; they never silently fall back to CPU. One run uses one GPU. See [device guidance](docs/DEVICE_GUIDE_ZH.txt).

## From raw CDS and proteins

The [Linux-first tutorial](docs/RAW_CDS_TUTORIAL.md) ([中文](docs/RAW_CDS_TUTORIAL_ZH.md)) includes synthetic CDS/PEP examples and a batch helper. It validates translations, performs MAFFT protein alignment and PAL2NAL codon back-translation, then runs `prepare`, `validate` and CPU fitting. Supply homologous pairs explicitly; KaKsTurbo does not infer orthology or repair frameshifts.

```bash
conda env create -p "$PWD/envs/input-prep" -f workflow/raw_cds_environment.yml
bash scripts/run_raw_cds_demo_linux.sh --output "$PWD/tutorial_demo"
```

These commands run from the **extracted Linux package**. Reuse an existing suitable alignment environment with `--env`; see the quick start. External aligners are optional preparation dependencies and are not embedded in the core binary.

## Commands and results

| Command | Purpose |
|---|---|
| `doctor` | Inspect runtime, CPU/GPU prerequisites and optional device probe |
| `prepare` | Normalize already aligned input and bind sequence metadata |
| `validate` | Check input syntax, pair identities and metadata |
| `run` | Fit all fourteen models and write model-averaged results |
| `status` | Read an existing run's committed progress |
| `resume` | Continue a compatible interrupted run with its original configuration |
| `export` | Optionally produce a KaKs_Calculator-compatible 22-column table |

The core files are `result.tsv` / `results.jsonl`; full model records, pair statistics, QC and `quality_summary.txt` / `quality_summary.json` remain available. Missing values remain NA/null. Terminal progress reports committed work, throughput, elapsed time, ETA and stage; `--quiet` suppresses it and `--json` keeps machine-readable output clean. HTML is not generated by default.

```bash
./kaksturbo export results_demo --format kaks-calculator --output compatible_demo
```

The optional export leaves the standard files intact. Missing or unvalidated statistics use NA with recorded reasons. Layout compatibility does not imply estimator equality. See [input/output](docs/INPUT_OUTPUT.md), [export mapping](docs/COMPATIBLE_EXPORT.txt), and [中文 CLI guide](USER_GUIDE_ZH.txt).

## Numerical interpretation

Finite estimates and passed coordinate checks do not establish convergence or a global optimum. Inspect model weights, boundary/cap flags and saved-endpoint projected gradients; the strict diagnostic threshold is 1e-8. A Ka/Ks ratio above one alone is not a significance test. Release installation tests are not new performance or biological-accuracy evidence. Existing checkpoints must use their matching release; identity checks are not bypassed.

## Source, license and support

Public Python interfaces, native C/C++/CUDA sources, build records, synthetic examples and tests are included. [Building native code](docs/BUILDING_NATIVE.md) explains the pinned-engine packaging workflow; native rebuilds require separate scientific qualification. Binary downloads and corresponding-source archives are provided together. PyPI and Bioconda publication are separate follow-up work; do not assume `pip install kaksturbo` or `conda install kaksturbo` selects this release.

License: [GPL-3.0-only](LICENSE). Retained dependencies keep their own [third-party notices](THIRD_PARTY_NOTICES.md). NVIDIA drivers and the CUDA Toolkit are external prerequisites for rebuilding and are not redistributed. The repository contains synthetic examples only, not private research data, manuscript files or benchmark intermediates.

Maintainer: **Taishan Li** · **litaishan@caf.ac.cn**. Report reproducible software issues through [GitHub Issues](https://github.com/707728642li/KaKsTurbo/issues); do not attach private sequences or credentials. No paper DOI is asserted here.
