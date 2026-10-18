<div align="center">

# KaKsTurbo

### Accelerated Ka/Ks estimation for comparative genomics

[![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20GPU-1b4353?style=flat-square)](docs/METHODS.md)
[![Precision](https://img.shields.io/badge/arithmetic-FP64-13858f?style=flat-square)](docs/METHODS.md)
[![Models](https://img.shields.io/badge/candidate%20models-14-13858f?style=flat-square)](docs/METHODS.md#complete-model-averaging)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-617681?style=flat-square)](LICENSE)

**[Download](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) · [Quick start](QUICKSTART.md) · [Methods](docs/METHODS.md) · [Benchmarks](docs/PERFORMANCE.md) · [中文](README_ZH.md)**

</div>

KaKsTurbo estimates nonsynonymous (**Ka**) and synonymous (**Ks**) substitution rates and their ratio from codon-aligned homologous gene pairs. It combines **complete fourteen-model averaging**, **structured FP64 matrix computation**, and **boundary-constrained optimization** with parallel CPU and batched GPU execution. Both backends share the same model definitions and numerical policies.

Each analysis retains candidate-model weights, optimization diagnostics, and explicit reasons for unavailable estimates. **Resumable runs and standardized TSV/JSONL results** support reproducible batch analyses, while the [CDS/PEP workflow](docs/RAW_CDS_TUTORIAL.md) connects raw sequence preparation to Ka/Ks estimation.

| Structured computation | Complete model averaging | Numerical diagnostics |
|:---|:---|:---|
| FP64 likelihood and derivative evaluation, persistent CPU workers, and batched GPU execution | Fourteen candidates and AICc-weighted parameters that reconstruct one averaged GTR process | Model weights, bound and iteration flags, projected gradients, and explicit NA reasons |

## Analysis workflow

[![KaKsTurbo workflow: codon filtering, structured CPU/GPU likelihoods, bounded optimization, complete model averaging and numerical follow-up.](docs/assets/workflow.png)](docs/assets/workflow.png)

**Workflow overview.** Codon-aligned gene pairs are processed through likelihood evaluation, bounded optimization, and complete model averaging. All 61 sense-codon states are retained; model weights and feasible-region drawings are schematic. Extended refitting is used in independent validation and is not part of a default CLI run. [Methods →](docs/METHODS.md)

The substitution-model families and model-averaging framework follow [established KaKs methods](docs/METHODS.md#statistical-foundations-and-baseline). KaKsTurbo focuses on their numerical implementation: structured matrix calculations, boundary-aware optimization, and parallel execution, supported by [explicit diagnostics and documented validation](docs/NUMERICAL_VALIDATION.md).

## Performance and scaling

[![Complete-workflow timings on a public 1,000-pair panel and archived plant/fly cohorts, with physical-core scaling on 10,000 pairs.](docs/assets/performance.png)](docs/assets/performance.png)

**Performance overview.** Complete CLI wall times and physical-core scaling. **A: N = 3; B/C: N = 5 technical runs.** Diamonds are medians; timing error bars show the observed minimum–maximum range. KAKS3 denotes **KaKs_Calculator 3.0, MA**. Each panel represents its stated dataset, build, and execution conditions.

| Dataset / timing cohort | N | KaKs_Calculator 3.0 | KaKsTurbo CPU | KaKsTurbo GPU | Ratio of median times |
|:---|---:|---:|---:|---:|:---|
| Public 1,000-pair panel · same Windows host | 3 | 458.04 s | **22.44 s** | **23.06 s** | **20.41× CPU · 19.86× GPU** |
| Plants · 29,028 pairs · archived Linux CPU cohort | 5 | 17,385.79 s | **1,520.30 s** | — | **11.44× CPU** |
| Flies · 23,571 pairs · archived Linux CPU cohort | 5 | 55,989.67 s | **1,221.29 s** | — | **45.84× CPU** |

These comparisons allocate **eight physical CPU cores / sixteen logical processors per arm**; the GPU arm additionally uses one GPU. The Linux cohort jobs used disjoint CPU allocations within a concurrent batch. On a separate fixed 10,000-pair scaling panel, 64 physical cores reached **22.12×** the two-core speed (**69.1% parallel efficiency**).

Speedups are ratios of median **end-to-end CLI times**, specific to these workloads and including estimator-policy differences from the baseline. GPU comparisons are limited to the same-host 1,000-pair panel, where CPU is slightly faster. [Timing protocol, all observations, and limitations →](docs/PERFORMANCE.md)

## Installation

### PyPI

Install into an existing Python environment:

```bash
python -m pip install --only-binary=:all: kaksturbo==1.1.0rc10
kaksturbo doctor --backend cpu
```

Requires Python ≥3.11 and Linux x86-64/glibc ≥2.17 or Windows x86-64. CPU execution needs no CUDA. [Package-index installation and qualification](docs/PACKAGE_INDEX.md).

### Portable packages

Download and extract the **whole portable package**. Both platforms include their runtime and CPU/GPU backends; core CPU execution needs neither Python, Conda nor CUDA installed separately.

| Platform | Program version | Download |
|:---|:---|:---|
| **Linux x86-64** · glibc ≥2.17 | 1.1.0rc9-tutorial1 | [Portable TAR.GZ](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc9-tutorial1-docs2-linux-x86_64.tar.gz) |
| **Windows x86-64** | 1.1.0rc10 | [Portable ZIP](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc10-docs2-windows-x86_64.zip) · [Extraction helper](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/Install-KaKsTurbo.ps1) |

[SHA256 checksums](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/SHA256SUMS-docs2.txt) · [Verified platforms](docs/LINUX_COMPATIBILITY.md) · [Windows long paths](docs/WINDOWS_PATHS.md)

<details>
<summary><strong>Package versions and offline documentation</strong></summary>

The downloads are **release candidates**. Windows rc10 includes path handling for the qualified interface; Linux retains its separately qualified rc9-tutorial1 portable build. The PyPI package is version **1.1.0rc10** on both platforms. A source checkout is intended for development.

The `docs2` archives include a bundled README, bilingual tutorials, methods/performance figures, and a synthetic result preview for offline reading. This suffix identifies a documentation-only revision; program versions are shown in the table above. Original packages remain available in the same Release.

</details>

**Bioconda:** the [recipe](https://github.com/bioconda/bioconda-recipes/pull/70404) awaits review and public-channel publication. Current installation routes are PyPI and the portable packages above.

## Quick start

The examples below use the portable package. For the PyPI installation, see the [installed-command quick start](docs/PACKAGE_INDEX.md#quick-start-with-the-installed-command).

### Codon-aligned input

After downloading the Linux archive:

```bash
tar -xzf KaKsTurbo-1.1.0rc9-tutorial1-docs2-linux-x86_64.tar.gz
cd KaKsTurbo-1.1.0rc9-tutorial1
./kaksturbo doctor --backend cpu
./kaksturbo run --input examples/three_pair_smoke.axt \
  --output results_demo --backend cpu --threads 8
head -n 4 results_demo/result.tsv
```

Use a new output directory. To analyze your own aligned AXT, replace the example input. For aligned FASTA, supply an explicit `--pairs pairs.tsv` manifest. [Input formats and metadata →](docs/INPUT_OUTPUT.md)

<details>
<summary><strong>Windows quick start</strong></summary>

Put the ZIP and extraction helper in the same folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-KaKsTurbo.ps1 -Archive .\KaKsTurbo-1.1.0rc10-docs2-windows-x86_64.zip -Destination "D:\My projects\KaKsTurbo"
& "D:\My projects\KaKsTurbo\kaksturbo.exe" doctor --backend cpu
& "D:\My projects\KaKsTurbo\kaksturbo.exe" run --input "D:\My projects\KaKsTurbo\examples\three_pair_smoke.axt" --output "D:\My results\run 1" --backend cpu --threads 8
```

Spaces, Unicode and long input/output paths are supported. For deep installation directories, use the PowerShell entry printed by the extractor. See [Windows path guidance](docs/WINDOWS_PATHS.md).

</details>

### Raw CDS and protein sequences

Provide `cds.fa`, `pep.fa` and an explicit list of homologous pairs. The [batch tutorial](docs/RAW_CDS_TUTORIAL.md) checks translation consistency, runs **MAFFT → PAL2NAL**, prepares codon-aligned inputs and launches KaKsTurbo. The extracted packages include the scripts and synthetic examples.

```bash
# Optional alignment environment; reuse an existing suitable environment if available.
conda env create -p "$PWD/envs/input-prep" -f workflow/raw_cds_environment.yml
bash scripts/run_raw_cds_demo_linux.sh --output "$PWD/tutorial_demo"
```

The preparation tools are separate from the core runtime. Homology assignment is supplied by the user; this workflow does not infer orthology or repair frameshifts. [English tutorial](docs/RAW_CDS_TUTORIAL.md) · [中文教程](docs/RAW_CDS_TUTORIAL_ZH.md)

### Execution modes and GPU configuration

| Mode | Backend | Use |
|:---|:---|:---|
| `robust` (default) | CPU or GPU | Complete model averaging with the shared bounded optimizer and diagnostics |
| `reproduce` | CPU | Historical estimator behavior for compatibility work; a distinct numerical policy |

For GPU execution, first check the device:

```bash
nvidia-smi
./kaksturbo doctor --backend gpu --probe-gpu
./kaksturbo run --input examples/three_pair_smoke.axt \
  --output results_gpu --backend gpu --device cuda:0
```

One run uses one NVIDIA GPU. The compiled backend requires compute capability **≥8.6**, driver CUDA compatibility **≥12.9**, and **≥8 GiB free memory**, including a 2 GiB reserve. A separate CUDA Toolkit is not needed. `auto` selects CPU; an explicit GPU request reports a device/driver/memory problem with corrective guidance instead of silently switching backends. [Device requirements](docs/DEVICES.md)

## Results and numerical diagnostics

Selected columns from the bundled three-pair example are shown below (**Windows CPU**, rounded to six decimals; Linux last digits may differ).

| pair_id | Ka | Ks | Ka_Ks | status |
|:---|---:|---:|---:|:---|
| forward | 0.066008 | 0.065604 | 1.006154 | available |
| reverse | 0.066008 | 0.065604 | 1.006154 | available |
| identical | NA | NA | NA | unidentifiable |

[Full-precision TSV](examples/results/three_pair_cpu.tsv) · [JSONL](examples/results/three_pair_cpu.jsonl) · [Quality summary](examples/results/three_pair_cpu_quality.txt)

The identical-sequence control has an undefined ratio and retains **NA** with its reason. The complete table also carries sequence IDs, codon counts, model selection, and boundary/cap diagnostics. `available` records result availability; convergence requires separate numerical assessment. [Output schema and interpretation →](docs/INPUT_OUTPUT.md)

| Output | Purpose |
|:---|:---|
| `result.tsv`, `results.jsonl` | Core results for downstream analysis |
| `models.tsv`, `models.jsonl` | Candidate-model fits, weights and diagnostics |
| `pair_statistics.jsonl`, `qc.tsv` | Retained-codon statistics and input QC |
| `quality_summary.txt`, `quality_summary.json` | Finite/NA coverage, caps, bounds and gradient flags |

Optional compatibility export preserves the standard results:

```bash
./kaksturbo export results_demo --format kaks-calculator --output compatible_demo
```

It provides a KaKs_Calculator-style 22-column table; unsupported or unvalidated statistics are **NA with reasons**. Matching a file layout does not imply estimator equivalence. [Field mapping and reader examples](docs/COMPATIBLE_EXPORT.txt)

## Workflow integration

- **Progress in the terminal:** committed pair count, throughput, elapsed time, ETA and processing stage.
- **Checkpoints:** `status results_demo` inspects progress; `resume results_demo` continues a compatible interrupted run.
- **Pipeline-friendly output:** `--json` for structured command output and `--quiet` to hide progress; no HTML is generated by default.
- **Input traceability:** directed pair IDs, order, byte hashes and prepared metadata remain linked to results.

## Documentation

| Start an analysis | Understand the calculation | Operate and extend |
|:---|:---|:---|
| [Quick start](QUICKSTART.md) / [中文](QUICKSTART_ZH.md) | [Methods](docs/METHODS.md) | [CLI reference](docs/USAGE.md) |
| [Raw CDS/PEP workflow](docs/RAW_CDS_TUTORIAL.md) | [Performance evidence](docs/PERFORMANCE.md) | [Devices and diagnostics](docs/DEVICES.md) |
| [Input and output](docs/INPUT_OUTPUT.md) | [Numerical interpretation](docs/NUMERICAL_VALIDATION.md) | [Source and build](docs/BUILDING_NATIVE.md) |
| [Linux compatibility](docs/LINUX_COMPATIBILITY.md) | [Reproducibility](docs/REPRODUCIBILITY.md) | [Version/checkpoint migration](docs/MIGRATION.md) |

## Scope and interpretation

KaKsTurbo supports **MA under standard genetic code 1**. Finite-coordinate acceptance, projected-gradient diagnostics, and independent reference fitting provide distinct levels of numerical evidence. CPU and GPU may return slightly different endpoints; a finite result does not establish a global optimum. **Ka/Ks > 1 alone is not a positive-selection significance test.** Known unresolved cases are documented in the [validation scope](docs/NUMERICAL_VALIDATION.md).

## Citation, license and support

Please cite the software version and [release URL](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10); machine-readable metadata are in [CITATION.cff](CITATION.cff). A paper DOI is not yet provided.

**GPL-3.0-only** · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Report an issue](https://github.com/707728642li/KaKsTurbo/issues)

Maintainer: **Taishan Li** · **litaishan@caf.ac.cn**. Please use a minimal synthetic example when reporting a problem.
