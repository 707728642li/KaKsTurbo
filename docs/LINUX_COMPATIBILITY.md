# Linux installation and compatibility

KaKsTurbo 1.1.0rc9 adds optional read-only compatibility export on Windows and Linux. Windows rc7 and Linux rc8 remain preserved. The native engines and bundled runtimes are byte-identical to those releases, so their numerical/ABI qualification below is inherited. New installed rc9 CLI/export checks are recorded separately in the release QUALIFICATION.json. No artifact has been uploaded.

## Portable Linux download

Extract the entire `KaKsTurbo-1.1.0rc9-linux-x86_64.tar.gz` archive and keep `runtime/` next to the executable:

```sh
tar -xzf KaKsTurbo-1.1.0rc9-linux-x86_64.tar.gz
cd KaKsTurbo-1.1.0rc9
./kaksturbo --version
./kaksturbo doctor --backend cpu
./kaksturbo run --input examples/three_pair_smoke.axt --output results --backend cpu --threads 2
./kaksturbo resume results
```

The private Python interpreter is included. Neither a system Python nor Conda is needed. This package targets x86-64 Linux with glibc >=2.17; it does not bundle or replace system glibc. ARM, macOS and musl/Alpine are not supported by this binary. The complete portable payload contains 95 ELF files, all requiring no symbol newer than glibc2.17. `auditwheel show` confirms the wheel's `manylinux_2_17_x86_64` tag. A symbol audit is an ABI check, not a promise that every distribution or driver works.

Inherited rc8 numerical tests (not rerun for the export-only rc9 update): native Ubuntu22.04/glibc2.35 CPU (1000 pairs plus49 hard cases), Ubuntu22.04/glibc2.35 container GPU userspace on the Ubuntu24.04 GPU VM (same1000+49), and CentOS7/glibc2.17 container CPU installation/small fit/resume. The GPU container mounts only NVIDIA devices and specific NVIDIA driver libraries, not host libc. It uses the host kernel and driver; this does not certify a native Ubuntu22 GPU kernel/driver installation. All nine scientific outputs of each full run are byte-identical to the previously independently audited reference for that backend. All original inputs/results remain unchanged. These are installation regressions, not new performance benchmarks.

## GPU

```sh
nvidia-smi
./kaksturbo doctor --backend gpu --probe-gpu
./kaksturbo run --input examples/three_pair_smoke.axt --output gpu_results --backend gpu --device cuda:0
```

CPU operation requires no NVIDIA software. GPU operation requires an NVIDIA driver supporting CUDA12.9 or newer, compute capability8.6 or newer, and at least8GiB free memory at admission (including a2GiB reserve). The frozen PTX is unchanged. A CUDA Toolkit installation is unnecessary. A recent driver may impose a higher OS/kernel requirement than the application's glibc2.17 ABI floor. Ubuntu22.04 GPU userspace has been actually exercised; older GPU systems are not certified merely because the package starts there.

`--backend auto` selects CPU. Explicit GPU requests never silently fall back: missing/hidden devices, unavailable or old drivers, unsupported architecture, invalid device selection and insufficient memory produce a reason and corrective action. `doctor` checks availability; it is not a fit or convergence proof. The verification includes actual GPU fitting, hidden-device/invalid-device errors and resume. Use terminal progress or JSON output; no web preview is needed.

## Python installation and historical Conda package

For a Python3.11+ environment, install the supplied Linux wheel offline:

```sh
python -m pip install --no-index --no-deps kaksturbo-1.1.0rc9-py3-none-manylinux_2_17_x86_64.whl
kaksturbo doctor --backend cpu
```

The rc9 source archive is a local release material. The previously qualified Bioconda/Conda build remains rc8 and does not include this export command; it is retained as an older artifact, not presented as a new rc9 download. Use the rc9 portable or wheel for compatible export. The older source archive and Bioconda recipe remain preserved. A local recipe check or container build is not approval by Bioconda, and the proposed GitHub download URL is not live. See the accompanying qualification file for the precise installation/build checks completed for each artifact.

## Numerical limits retained

Fourteen models, FP64, parameter bounds,500 iterations, finite-coordinate gates and projected-gradient threshold1e-8 are unchanged. The1000-pair regressions retain14NA; the49-pair regression retains0NA and4 models above the strict gradient threshold. CPU and GPU have their pre-existing distinct gradient-flag counts; byte identity is asserted within each backend, not between CPU and GPU. A finite Ka/Ks or reproducible output is not a convergence/global-optimum certificate.
