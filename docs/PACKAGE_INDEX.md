# Package-index installation

## Availability

| Channel | Version | Status |
|---|---|---|
| [PyPI](https://pypi.org/project/kaksturbo/1.1.0rc10/) | 1.1.0rc10 | Published: Linux/Windows wheels and source archive |
| [Bioconda recipe PR #70404](https://github.com/bioconda/bioconda-recipes/pull/70404) | 1.1.0rc10 | Submitted; review, merge and public-channel upload pending |
| [GitHub portable downloads](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) | Windows rc10 / Linux rc9-tutorial1 | Existing original and docs1 archives retained |

These versions are release candidates. The PyPI wheel uses your Python environment; a portable archive supplies its own Python runtime. Historical portable documentation is a snapshot, so consult this page for current channel availability. Do not mix old checkpoints with a different release.

## Quick start with the installed command

Requires Python ≥3.11, Linux x86-64/glibc ≥2.17 or Windows x86-64. ARM and macOS native packages are not supplied. CPU execution needs no CUDA installation.

```bash
python -m pip install --only-binary=:all: kaksturbo==1.1.0rc10
kaksturbo --version
kaksturbo doctor --backend cpu
curl -fL https://raw.githubusercontent.com/707728642li/KaKsTurbo/v1.1.0rc10/examples/three_pair_smoke.axt -o demo.axt
kaksturbo run --input demo.axt --output demo_results --backend cpu --threads 2
```

The fixed-release synthetic input has three requests: two finite ratios and one NA. Windows PowerShell users should use `curl.exe` in place of `curl`. Use new output directories and avoid overwriting an existing `demo.axt`. Replace `--input` with your own codon-aligned AXT for actual analysis. `kaksturbo examples` also generates small format templates, whose Ka/Ks can be unidentifiable; they are not this three-pair numerical demonstration. [Input/output schema](INPUT_OUTPUT.md)

For an isolated Conda environment, reuse a suitable existing environment or create a project-local prefix:

```bash
conda create -p ./envs/kaksturbo python=3.11 pip
conda run -p ./envs/kaksturbo python -m pip install --only-binary=:all: kaksturbo==1.1.0rc10
conda run -p ./envs/kaksturbo kaksturbo doctor --backend cpu
```

This installs the **PyPI wheel into a Conda environment**, not a package from the Bioconda channel. A `conda install -c bioconda kaksturbo` command will be advertised only after actual channel publication.

## GPU and alignment tools

Run `nvidia-smi`, then `kaksturbo doctor --backend gpu --probe-gpu` before explicitly selecting `--backend gpu --device cuda:0`. The native backend requires compute capability ≥8.6, driver CUDA compatibility ≥12.9 and at least 8 GiB free device memory, including a 2 GiB reserve. No separate CUDA Toolkit is required. An unavailable/incompatible GPU produces a diagnostic error; an explicit GPU request never silently becomes a CPU run. [Device guidance](DEVICES.md)

The pip wheel installs the calculator and pinned engine. The full upstream MAFFT/PAL2NAL tutorial, preparation helpers and examples are provided in the repository and portable archives. When using the repository's [raw-CDS demo](RAW_CDS_TUTORIAL.md), set `KAKSTURBO_BIN` to the absolute installed command and pass `--env` for your existing alignment environment. Homology assignment and codon alignment remain explicit prerequisites.

## Installation evidence and limits

- Windows Python 3.11: wheel installed by pip; installed CLI passed CPU doctor, validation, the three-pair example, resume and compatible export. Its result table matched the published Windows synthetic preview.
- Native Ubuntu 22.04.5: isolated wheel installation retained all 1,000 + 49 requests, 14,686 model slots and respectively 14 + 0 MA NA. All nine standard scientific files and model-quality counts matched the corresponding previously qualified CPU references; version metadata is compared separately.
- Fresh installation **from PyPI** passed the three-pair example (2 finite +1 NA), the raw-CDS tutorial (3 finite +1 NA), resume/export and clear no-GPU failures.
- Bioconda candidate: native libraries rebuilt with the official pinned cos7 build container, Conda C++ compiler and glibc 2.17 sysroot. Recipe tests and all lint checks passed. Installing the resulting Conda artifact on Ubuntu 22.04 passed the same 1,000 +49 pair regression and raw-CDS tutorial. This is local package qualification, not evidence that the package is already in the public channel.
- Upstream [Bioconda CI](https://github.com/bioconda/bioconda-recipes/actions/runs/39115068076) passed native Linux builds, three-pair CPU recipe tests and artifact collection for Python 3.11, 3.12 and 3.13. The 1,000 +49 pair numerical qualification above was run on Python 3.11; it is not claimed for every Python version. Upstream review and public-channel upload remain pending.
- Wheel ABI inspection confirms `manylinux_2_17_x86_64`. That compatibility target does not claim a new full numerical run on every glibc-2.17 distribution.
- Existing GPU numerical qualification is retained through the unchanged pinned engine; this index-release work did not rerun GPU fitting. Four hard-panel model fits remain above the strict projected-gradient threshold. Byte identity proves regression consistency, not full convergence or a global optimum.

All fourteen models, FP64, 500 iterations, boundaries and the 1e-8 diagnostic threshold are unchanged. The paper and original results are untouched. [Machine-readable summary](../packaging/index_qualification.json)

## Source archives and license metadata

The first source upload was rejected because `PKG-INFO` used wheel-relative license paths. The corrected source uses actual source-archive paths, and all 103 declared license paths are verified. Both wheels have 99 valid declared paths; they were not replaced. The corrected source is published on PyPI as `kaksturbo-1.1.0rc10.tar.gz` and retained on GitHub as `kaksturbo-1.1.0rc10-sdist1.tar.gz` with identical bytes. Use [SHA256SUMS-indexes1.txt](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/SHA256SUMS-indexes1.txt). Older archives and checksums remain historical records.

The GitHub publication workflow is manually dispatched from `main`, uses the dedicated `pypi` environment, fetches hash-pinned qualified distributions and publishes using PyPI Trusted Publishing. No long-lived PyPI token is stored in the repository.
