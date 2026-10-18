<div align="center">

# KaKsTurbo

### High-performance, model-averaged Ka/Ks estimation on CPUs and GPUs

[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-1b4353?style=flat-square)](#installation)
[![Precision](https://img.shields.io/badge/arithmetic-FP64-13858f?style=flat-square)](docs/METHODS.md)
[![Models](https://img.shields.io/badge/candidate%20models-14-13858f?style=flat-square)](docs/METHODS.md#complete-model-averaging)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-617681?style=flat-square)](LICENSE)

**[Download](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) · [Quick start](QUICKSTART.md) · [Methods](docs/METHODS.md) · [Benchmarks](docs/PERFORMANCE.md) · [中文](README_ZH.md)**

</div>

KaKsTurbo estimates nonsynonymous and synonymous substitution rates (**Ka**, **Ks**) and their ratio for batches of homologous, codon-aligned gene pairs. It brings structured FP64 matrix computation and boundary-constrained optimization to a complete fourteen-model averaging workflow, with CPU and single-GPU backends that share the same model definitions and numerical policies.

Built for comparative-genomics workflows, KaKsTurbo combines high-throughput execution with **model-level diagnostics, resumable jobs, and analysis-ready TSV/JSONL results**. A Linux-first tutorial takes raw CDS and protein sequences through alignment to the standard input format.

| Compute efficiently | Retain the model average | Inspect every estimate |
|:---|:---|:---|
| Persistent CPU workers, batched GPU evaluation and reused workspaces | All fourteen candidates, FP64 arithmetic and one reconstructed averaged GTR process | Saved model weights, boundary/iteration flags, projected gradients and explicit NA reasons |

## From aligned genes to interpretable results

[![KaKsTurbo workflow: codon filtering, structured CPU/GPU likelihoods, bounded optimization, complete model averaging and numerical follow-up.](docs/assets/workflow.png)](docs/assets/workflow.png)

**Figure 1.** The shared numerical workflow and its implementation. All 61 sense-codon states are retained. Model weights and feasible-region drawings are schematic. Independent extended fitting is a validation procedure, not an automatic step in every production run. [Explore the method →](docs/METHODS.md)

The statistical model families and model-averaging framework build on [established KaKs methods](docs/METHODS.md#statistical-foundations-and-baseline). KaKsTurbo's work focuses on their computational implementation, boundary-aware optimization and transparent numerical validation; it does not introduce a new biological selection test.

## Measured performance

[![Complete-workflow timings on a public 1,000-pair panel and archived plant/fly cohorts, with physical-core scaling on 10,000 pairs.](docs/assets/performance.png)](docs/assets/performance.png)

**Figure 2.** Complete CLI wall times and physical-core scaling. **A: N = 3; B/C: N = 5 technical runs.** Diamonds are medians; timing error bars show the full observed minimum–maximum range. KAKS3 denotes **KaKs_Calculator 3.0, MA**. The panels retain their distinct datasets, builds and execution conditions.

| Dataset / timing cohort | N | KaKs_Calculator 3.0 | KaKsTurbo CPU | KaKsTurbo GPU | Ratio of median times |
|:---|---:|---:|---:|---:|:---|
| Public 1,000-pair panel · same Windows host | 3 | 458.04 s | **22.44 s** | **23.06 s** | **20.41× CPU · 19.86× GPU** |
| Plants · 29,028 pairs · archived Linux CPU cohort | 5 | 17,385.79 s | **1,520.30 s** | — | **11.44× CPU** |
| Flies · 23,571 pairs · archived Linux CPU cohort | 5 | 55,989.67 s | **1,221.29 s** | — | **45.84× CPU** |

These comparisons allocate **eight physical CPU cores / sixteen logical processors per arm**; the GPU arm additionally uses one GPU. The Linux cohort jobs used disjoint CPU allocations within a concurrent batch. On a separate fixed 10,000-pair scaling panel, 64 physical cores reached **22.12×** the two-core speed (**69.1% parallel efficiency**).

These are workload-specific **complete-workflow comparisons**, including estimator-policy differences from the baseline. They are not a universal speedup or an assertion that CPU and GPU results are bitwise identical. The GPU is slightly slower than the CPU on the same-host 1,000-pair panel. No cross-platform GPU speedup is claimed. [Timing protocol, all observations and limitations →](docs/PERFORMANCE.md)

## Installation

Download and extract the **whole portable package**. Both platforms include their runtime and CPU/GPU backends; core CPU execution needs neither Python, Conda nor CUDA installed separately.

| Platform | Program version | Download |
|:---|:---|:---|
| **Linux x86-64** · glibc ≥2.17 | 1.1.0rc9-tutorial1 | [Portable TAR.GZ](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc9-tutorial1-docs1-linux-x86_64.tar.gz) |
| **Windows x86-64** | 1.1.0rc10 | [Portable ZIP](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc10-docs1-windows-x86_64.zip) · [Extraction helper](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/Install-KaKsTurbo.ps1) |

[SHA256 checksums](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/SHA256SUMS-docs1.txt) · [Verified platforms](docs/LINUX_COMPATIBILITY.md) · [Windows long paths](docs/WINDOWS_PATHS.md)

The `docs1` downloads include this README, bilingual tutorials, methods/performance figures and the synthetic result preview for offline reading. `docs1` is a documentation-only archive revision: the Windows and Linux program versions below remain unchanged, and the original packages remain available in the same Release.

The downloads are **release candidates**. Windows rc10 adds path handling to the qualified interface; Linux retains its separately qualified rc9-tutorial1 build. A source checkout is for development and is not a stand-alone installation. PyPI **1.1.0rc10** is now available for Linux and Windows; the separately qualified portable versions above remain unchanged. The [Bioconda recipe](https://github.com/bioconda/bioconda-recipes/pull/70404) is submitted and awaits review/channel publication. [Package-index installation and qualification](docs/PACKAGE_INDEX.md).

### Install into an existing Python environment

```bash
python -m pip install --only-binary=:all: kaksturbo==1.1.0rc10
kaksturbo doctor --backend cpu
```

Requires Python ≥3.11 and Linux x86-64/glibc ≥2.17 or Windows x86-64. CPU use needs no CUDA. See the [pip quick start](docs/PACKAGE_INDEX.md#quick-start-with-the-installed-command); no working Bioconda-channel installation is claimed until the recipe is merged and uploaded.

## Run your first analysis

### Already have codon-aligned sequences?

After downloading the Linux archive:

```bash
tar -xzf KaKsTurbo-1.1.0rc9-tutorial1-docs1-linux-x86_64.tar.gz
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
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-KaKsTurbo.ps1 -Archive .\KaKsTurbo-1.1.0rc10-docs1-windows-x86_64.zip -Destination "D:\My projects\KaKsTurbo"
& "D:\My projects\KaKsTurbo\kaksturbo.exe" doctor --backend cpu
& "D:\My projects\KaKsTurbo\kaksturbo.exe" run --input "D:\My projects\KaKsTurbo\examples\three_pair_smoke.axt" --output "D:\My results\run 1" --backend cpu --threads 8
```

Spaces, Unicode and long input/output paths are supported. For deep installation directories, use the PowerShell entry printed by the extractor. See [Windows path guidance](docs/WINDOWS_PATHS.md).

</details>

### Starting from raw CDS and proteins?

Provide `cds.fa`, `pep.fa` and an explicit list of homologous pairs. The [batch tutorial](docs/RAW_CDS_TUTORIAL.md) checks translation consistency, runs **MAFFT → PAL2NAL**, prepares codon-aligned inputs and launches KaKsTurbo. The extracted packages include the scripts and synthetic examples.

```bash
# Optional alignment environment; reuse an existing suitable environment if available.
conda env create -p "$PWD/envs/input-prep" -f workflow/raw_cds_environment.yml
bash scripts/run_raw_cds_demo_linux.sh --output "$PWD/tutorial_demo"
```

The preparation tools are separate from the core runtime. Homology assignment is supplied by the user; this workflow does not infer orthology or repair frameshifts. [English tutorial](docs/RAW_CDS_TUTORIAL.md) · [中文教程](docs/RAW_CDS_TUTORIAL_ZH.md)

### Select CPU, GPU or historical reproduction

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

## Results you can use

The example above produces the following selected columns. These are **actual Windows CPU results from the bundled synthetic input**, rounded here to six decimal places; Linux last digits may differ.

| pair_id | Ka | Ks | Ka_Ks | status |
|:---|---:|---:|---:|:---|
| forward | 0.066008 | 0.065604 | 1.006154 | available |
| reverse | 0.066008 | 0.065604 | 1.006154 | available |
| identical | NA | NA | NA | unidentifiable |

[Full-precision TSV](examples/results/three_pair_cpu.tsv) · [JSONL](examples/results/three_pair_cpu.jsonl) · [Quality summary](examples/results/three_pair_cpu_quality.txt)

The identical-sequence control retains **NA**, with its reason, rather than an invented zero ratio. `available` describes result availability, not a convergence certificate. The complete table also carries sequence IDs, codon counts, model selection and boundary/cap diagnostics. [Output schema and interpretation →](docs/INPUT_OUTPUT.md)

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

## Built for batch workflows

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

KaKsTurbo currently supports **MA under standard genetic code 1**. Finite-coordinate acceptance, projected-gradient diagnostics and independent reference fitting answer different questions. A finite result does not establish a global optimum, and Ka/Ks > 1 alone is not a positive-selection significance test. Known unresolved cases are retained in the [validation scope](docs/NUMERICAL_VALIDATION.md).

## Citation, license and support

Please cite the software version and [release URL](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10); machine-readable metadata are in [CITATION.cff](CITATION.cff). A paper DOI is not yet provided.

**GPL-3.0-only** · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Report an issue](https://github.com/707728642li/KaKsTurbo/issues)

Maintainer: **Taishan Li** · **litaishan@caf.ac.cn**. Please use a minimal synthetic example when reporting a problem.
