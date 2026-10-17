# Start here: raw CDS to Ka/Ks

[中文快速入门](QUICKSTART_ZH.md) · [Full tutorial](docs/RAW_CDS_TUTORIAL.md) · [Batch helper](scripts/prepare_cds_pairs.py)

Extract the **whole Linux portable package** into a new directory and open a terminal there. CPU calculation needs no GPU or CUDA installation. The bundled Python runs KaKsTurbo; the optional alignment environment below runs MAFFT/PAL2NAL.

## Linux: install optional alignment tools once

If you already have suitable tools, reuse that Conda environment with `--env` instead of installing again. Otherwise, from the extracted directory:

```bash
conda env create -p "$PWD/envs/input-prep" -f workflow/raw_cds_environment.yml
```

The recipe uses conda-forge/bioconda and pins the tested MAFFT 7.526 and PAL2NAL 14.1 interfaces. Initial installation requires access to those channels; the aligners are not embedded in the binary package. Python and GNU parallel are included in this optional environment. The helper does not change the scientific engine.

## Linux: run the complete four-request demo

```bash
bash scripts/run_raw_cds_demo_linux.sh --output "$PWD/tutorial_demo"
```

This runs CPU diagnostics, raw CDS/PEP checks, parallel protein alignment, codon back-translation, `prepare`, `validate` and CPU fitting. It uses two single-threaded alignment jobs, followed by two CPU fitting threads. Inspect `tutorial_demo/results_cpu/result.tsv` and `quality_summary.txt`. The synthetic fixture has four directed requests, including a reverse request and an identical-sequence control; the tested result has three finite Ka/Ks and one NA. It is not a benchmark dataset.

For an existing environment or GNU parallel installed elsewhere:

```bash
bash scripts/run_raw_cds_demo_linux.sh \
  --env /absolute/path/to/existing/input-prep \
  --parallel /absolute/path/to/parallel \
  --output "$PWD/tutorial_demo_existing_env"
```

Each output directory must be new; a failure retains its evidence and stops subsequent steps. `--jobs` and `--threads` control the two sequential stages separately. For the source tree, set `KAKSTURBO_BIN=/absolute/path/to/installed/kaksturbo` first. See `--help` for options. The demo wrapper runs CPU only.

## Use your own data

Provide matching `cds.fa`, `pep.fa` and an explicit TAB-separated `pairs.tsv` with columns `pair_id`, `seq1`, `seq2`. Decide homologous relationships before alignment. Then follow the [full batch tutorial](docs/RAW_CDS_TUTORIAL.md#use-your-own-data-in-parallel). It explains ID mapping, terminal-stop opt-in, retained gap codons, failures and CPU/GPU commands. The basic helper requires frame-intact genetic-code-1 CDS and matching proteins; it does not repair frameshifts or infer orthology.

## Windows

The Windows package contains the same tutorial, script and raw example. Prepare the inputs on Linux as above, then copy the **whole** `tutorial_demo/prepared/` directory to Windows. Windows rc10 handles long input/output paths, spaces and Unicode without changing the system long-path policy. See docs/WINDOWS_PATHS.md for deep-installation launch and safe extraction. The scientific engine is unchanged from rc9. From the Windows package directory:

```powershell
.\kaksturbo.exe validate --input prepared
.\kaksturbo.exe run --input prepared --output D:/kt_run1 --backend cpu --threads 2
```

For an immediate Windows example without alignment software, use `examples/three_pair_smoke.axt` as `--input`. Native Windows MAFFT/PAL2NAL installation is not qualified by this tutorial; no WSL setup is required or performed.

Explicit GPU use requires a supported NVIDIA driver/device and sufficient free memory. Check `doctor --backend gpu --probe-gpu` before selecting `--backend gpu`; see [device requirements](docs/DEVICE_GUIDE_ZH.txt) and [Linux validation scope](docs/LINUX_COMPATIBILITY.md). GPU failures are reported rather than silently switching to CPU.
