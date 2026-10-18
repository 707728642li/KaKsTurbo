# KaKsTurbo

**Model-averaged Ka/Ks estimation with CPU/GPU execution and explicit numerical diagnostics.**

[Source and documentation](https://github.com/707728642li/KaKsTurbo) · [Quick start](https://github.com/707728642li/KaKsTurbo/blob/main/QUICKSTART.md) · [Raw CDS/PEP tutorial](https://github.com/707728642li/KaKsTurbo/blob/main/docs/RAW_CDS_TUTORIAL.md) · [Portable downloads](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10)

KaKsTurbo estimates nonsynonymous and synonymous substitution rates (Ka and Ks), and their ratio, for batches of homologous codon-aligned gene pairs. It combines structured FP64 matrix computation, boundary-constrained optimization and complete fourteen-model averaging. The statistical model families and averaging framework build on established KaKs methods; this is a computational implementation with numerical validation, not a new biological selection test.

![KaKsTurbo workflow](https://raw.githubusercontent.com/707728642li/KaKsTurbo/main/docs/assets/workflow.png)

## Installation

This is a **release candidate**, version **1.1.0rc10**. Use Python **3.11 or later** on **Linux x86-64 (glibc >=2.17)** or **Windows x86-64**. ARM and macOS are not supported by these native wheels.

```bash
python -m pip install --only-binary=:all: kaksturbo==1.1.0rc10
kaksturbo --version
kaksturbo doctor --backend cpu
```

The wheel includes the pinned native CPU/GPU engine. CPU execution needs no CUDA installation. Existing portable GitHub downloads include their own Python runtime and remain available separately; this pip distribution uses the Python environment into which it is installed.

## Analyze aligned pairs

```bash
kaksturbo validate --input aligned_pairs.axt
kaksturbo run --input aligned_pairs.axt --output results_cpu --backend cpu --threads 8
kaksturbo status results_cpu
```

Supply codon-aligned AXT, or aligned FASTA plus an explicit pair manifest. Raw CDS and protein files require alignment first: the [Linux-first batch tutorial](https://github.com/707728642li/KaKsTurbo/blob/main/docs/RAW_CDS_TUTORIAL.md) provides MAFFT/PAL2NAL preparation scripts, examples and validation. Download those companion files from the repository or portable archive; they are not installed as commands by pip.

Use a new output directory. `kaksturbo resume results_cpu` continues a compatible interrupted run. Terminal progress reports committed pairs and processing stages; `--json` provides machine-readable command output and `--quiet` suppresses progress.

## GPU requirements

```bash
nvidia-smi
kaksturbo doctor --backend gpu --probe-gpu
kaksturbo run --input aligned_pairs.axt --output results_gpu --backend gpu --device cuda:0
```

One run uses one NVIDIA GPU. The compiled backend requires compute capability **>=8.6**, driver CUDA compatibility **>=12.9** and **>=8 GiB free memory**, including a 2 GiB reserve. A separate CUDA Toolkit is not needed. `auto` selects CPU. An explicit GPU request reports unavailable devices, incompatible drivers or insufficient memory with corrective guidance; it never silently falls back to CPU. See [device requirements](https://github.com/707728642li/KaKsTurbo/blob/main/docs/DEVICES.md).

## Outputs and interpretation

| Files | Contents |
|---|---|
| `result.tsv`, `results.jsonl` | Ka, Ks, Ka/Ks, pair identifiers and availability |
| `models.tsv`, `models.jsonl` | All fourteen model slots, weights and diagnostics |
| `pair_statistics.jsonl`, `qc.tsv` | Codon statistics and input quality |
| `quality_summary.txt`, `quality_summary.json` | Finite/NA coverage, boundaries, iteration caps and gradient flags |

Missing or unidentifiable estimates remain **NA with reasons**. Finite output and optimizer success are not convergence certificates. The projected-gradient diagnostic threshold remains **1e-8**; unresolved numerical cases are documented. Ka/Ks >1 alone is not a significance test for positive selection. Current scope is **MA under standard genetic code 1**, with FP64 and all fourteen models preserved.

Optional compatibility export writes a KaKs_Calculator-style table without replacing the standard results:

```bash
kaksturbo export results_cpu --format kaks-calculator --output compatible_results
```

Unsupported or unvalidated statistics are NA with explanations. File-format compatibility does not imply estimator equivalence.

## Performance, methods and support

The [performance report](https://github.com/707728642li/KaKsTurbo/blob/main/docs/PERFORMANCE.md) contains measured full-workflow comparisons, replicate counts, physical-core allocations and limitations. Speedups depend on sequence characteristics and estimator policies; no universal acceleration or cross-platform GPU ratio is claimed.

[Methods](https://github.com/707728642li/KaKsTurbo/blob/main/docs/METHODS.md) · [Numerical validation](https://github.com/707728642li/KaKsTurbo/blob/main/docs/NUMERICAL_VALIDATION.md) · [Input/output schema](https://github.com/707728642li/KaKsTurbo/blob/main/docs/INPUT_OUTPUT.md) · [Build and corresponding source](https://github.com/707728642li/KaKsTurbo/blob/main/docs/BUILDING_NATIVE.md)

**GPL-3.0-only**; third-party license notices are included. Maintainer: **Taishan Li**, litaishan@caf.ac.cn. [Report issues](https://github.com/707728642li/KaKsTurbo/issues) using minimal synthetic inputs. Cite the version and release URL; no paper DOI is currently assigned.
