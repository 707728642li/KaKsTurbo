# CPU and GPU requirements

[Install](../README.md#installation) · [Linux platform scope](LINUX_COMPATIBILITY.md) · [中文设备说明](DEVICE_GUIDE_ZH.txt)

## CPU: the default path

The portable packages include their Python runtime and CPU libraries. CPU calculation requires no NVIDIA hardware, driver or CUDA installation. `--backend auto` currently selects CPU. `--threads` controls workers; the default is at most 16 CPUs available to the process.

```bash
./kaksturbo doctor --backend cpu
./kaksturbo run --input examples/three_pair_smoke.axt --output cpu_result --backend cpu --threads 8
```

## GPU: explicit single-device execution

| Requirement | Precompiled backend |
|:---|:---|
| Hardware | NVIDIA GPU, compute capability ≥8.6 |
| Driver | CUDA compatibility ≥12.9 |
| Free memory at admission | ≥8 GiB, including a 2 GiB reserve; actual use depends on workload |
| CUDA Toolkit | Not required for portable execution; required for native rebuilding |
| Devices per run | One; select by `cuda:N` or physical GPU UUID |

```bash
nvidia-smi
./kaksturbo doctor --backend gpu --probe-gpu
./kaksturbo run --input examples/three_pair_smoke.axt --output gpu_result --backend gpu --device cuda:0
```

`doctor` explains device selection and admission. `--probe-gpu` also initializes the driver; it does not allocate a fitting workload or certify convergence. A recent NVIDIA driver can require a newer host OS/kernel than the application's glibc floor.

## When a GPU cannot be used

An explicit GPU request fails with the recorded reason and corrective guidance; it does not silently run on CPU.

| Condition | Action |
|:---|:---|
| No NVIDIA device or hidden device | Inspect `nvidia-smi` and device visibility; select CPU if GPU use is not intended |
| Missing/old/incompatible driver | Install or update a supported NVIDIA driver, then repeat the diagnostic |
| Unsupported compute capability | Use supported hardware or the CPU backend |
| Invalid device index/UUID | Select a device listed by the diagnostic |
| Insufficient free memory | Wait for capacity or choose another eligible device; do not stop unrelated work |
| Missing/mismatched package assets | Re-extract the full verified package; do not move only its executable |

Windows supports verified local long/Unicode paths; [the extraction helper](WINDOWS_PATHS.md) handles deep installations. Linux supports the tested x86-64/glibc combinations listed in [platform qualification](LINUX_COMPATIBILITY.md). These checks diagnose installation and resource readiness, not numerical accuracy.
