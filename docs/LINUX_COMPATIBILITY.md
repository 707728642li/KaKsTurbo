# Linux installation and compatibility

Linux is the primary tutorial platform. The public portable package is **1.1.0rc9-tutorial1**, available from the [GitHub Release](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10). The release tag also carries Windows rc10; the separately qualified Linux executable is intentionally not relabeled rc10.

## Install the portable package

Download [the Linux TAR.GZ](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc9-tutorial1-docs1-linux-x86_64.tar.gz) and check it against [SHA256SUMS](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/SHA256SUMS-docs1.txt). Extract the entire archive:

```bash
tar -xzf KaKsTurbo-1.1.0rc9-tutorial1-docs1-linux-x86_64.tar.gz
cd KaKsTurbo-1.1.0rc9-tutorial1
./kaksturbo --version
./kaksturbo doctor --backend cpu
./kaksturbo run --input examples/three_pair_smoke.axt --output results --backend cpu --threads 2
./kaksturbo resume results
```

Keep `runtime/` beside the executable. No system Python or Conda is needed for core execution. The optional [raw-CDS preparation tutorial](RAW_CDS_TUTORIAL.md) uses a separate Conda environment for external aligners.

## ABI target and verified systems

The binary targets **x86-64 Linux with glibc ≥2.17** and does not replace system glibc. All 95 ELF files in the qualified runtime require no symbol newer than glibc 2.17. ARM, macOS and musl/Alpine are not supported by this binary. The ABI floor is not blanket certification of every distribution or NVIDIA driver.

| Environment | Verified scope |
|:---|:---|
| Native Ubuntu 22.04, glibc 2.35 | CPU installation and 1,000-pair / 49-pair numerical regression |
| Ubuntu 22.04 container on Ubuntu 24.04 NVIDIA host | GPU installation and the same 1,000-pair / 49-pair backend-specific regression |
| CentOS 7 container, glibc 2.17 | CPU installation, small fit and resume smoke check |
| Native Ubuntu 22.04.5 | Raw CDS/PEP → MAFFT/PAL2NAL → prepare → CPU fit on four synthetic requests |

The GPU container uses the host kernel and NVIDIA driver; it mounts driver libraries rather than host libc. It does not certify every native Ubuntu 22.04 kernel/driver combination. Each full regression's nine standard outputs matched its previously validated **same-backend** reference byte-for-byte. These are installation regressions, not new performance measurements.

## GPU readiness

```bash
nvidia-smi
./kaksturbo doctor --backend gpu --probe-gpu
./kaksturbo run --input examples/three_pair_smoke.axt --output gpu_results --backend gpu --device cuda:0
```

The precompiled backend requires compute capability ≥8.6, NVIDIA driver CUDA compatibility ≥12.9 and at least 8 GiB free GPU memory, including its reserve. No separate CUDA Toolkit installation is needed. Explicit GPU requests report unavailable devices, unsupported drivers/architecture or insufficient memory; they do not silently fall back to CPU. `auto` selects CPU. [Device guidance](DEVICES.md)

## Distribution channels and source

The **portable archive is the public ready-to-run download**. Corresponding source is attached to the Release; the main repository includes the Windows rc10 interface snapshot and the pinned platform-native source. A source checkout needs a version-matched engine for packaging. See [source layout](SOURCE_LAYOUT.md) and [native build guidance](BUILDING_NATIVE.md).

PyPI now provides **1.1.0rc10** native Linux/Windows wheels. The Linux wheel is tagged manylinux_2_17_x86_64; auditwheel confirms that ABI requirement. A fresh installation from PyPI on native Ubuntu 22.04.5 passed the CPU example and complete raw-CDS tutorial. The version-matched local wheel and independently source-built Conda package also passed the 1,000 + 49 pair CPU regression. Bioconda remains [under review](https://github.com/bioconda/bioconda-recipes/pull/70404), so local Conda verification must not be confused with public-channel availability. [Commands and exact scope](PACKAGE_INDEX.md). GPU numerical qualification is inherited through the unchanged pinned engine; the index-release tests did not rerun GPU fitting.

## Numerical limits retained

The earlier 1,000-pair installation panel retains 14 NA; the 49-pair hard panel retains zero MA NA and four model fits above the strict gradient threshold. These are different inputs from the newer public 1,000-pair timing panel. CPU/GPU diagnostic differences remain recorded. Finite or repeated output does not prove convergence or global optimality. [Numerical validation](NUMERICAL_VALIDATION.md)
