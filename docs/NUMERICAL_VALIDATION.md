# Numerical diagnostics and validation scope

[Methods](METHODS.md) · [Output fields](INPUT_OUTPUT.md) · [Performance](PERFORMANCE.md)

KaKsTurbo separates result availability, production acceptance, stationarity diagnostics and independent reference fitting. They should not be collapsed into a single claim of “accuracy”.

| Observation | Meaning | What it does not establish |
|:---|:---|:---|
| Finite Ka/Ks / `available` | A result could be constructed under the estimator's rules | Stationarity, global optimality or biological significance |
| Finite-coordinate gate passed | The prescribed nearby coordinate probes did not find an improvement beyond tolerance | The strict gradient condition |
| Projected gradient ≤1e-8 | The saved point meets this first-order diagnostic | A global optimum |
| Iteration cap / bound hit | A specific optimization event was recorded | That the final weighted estimate must be wrong |
| Byte-identical repeat | The recorded output is reproducible | That its endpoint is converged |

## Production diagnostic

The gradient is evaluated for `F/L` with respect to log parameters, where F is negative log likelihood and L is retained codon count. Projection accounts for the box constraints; the diagnostic compares its infinity norm with **1e-8**. Missing diagnostics remain unknown, distinct from a pass or a failure.

The finite-coordinate gate checks prescribed log-coordinate perturbations and uses its own `1e-6` raw-objective tolerance. Optimizer success alone is not used as a stationarity certificate.

## Independent sensitivity checks

The established follow-up checks the saved point, then fits from three fixed starts and the saved endpoint, with an extended budget. It compares objective values, analytic/finite-difference derivatives, model weights, the reconstructed full MA estimate, NA changes and crossings of Ka/Ks=1. Best-finite endpoints and strict-qualified reference endpoints are retained separately. Missing one of fourteen strict model endpoints leaves the complete strict reference unresolved.

For interpretation, a high-weight unstable model deserves particular attention, but low weight alone does not justify deleting a candidate. The same complete model average must be rebuilt before assessing the final Ka/Ks change.

## Retained limits in the released software

- The 49-pair hard-panel installation regression retains **four model fits above the strict gradient threshold**. These are not relabeled as converged.
- The separate public 1,000-pair timing panel retains 353 flagged CPU model slots and 254 flagged GPU model slots in each technical repeat; its 1,000 MA results are finite. These are different inputs from the earlier installation panel with 14 NA.
- CPU and GPU can have slightly different endpoints and gradient flags. Same-backend repeat identity is not a claim of CPU/GPU bitwise equality.
- Zero substitutions and insufficient information can make a ratio undefined. NA remains explicit.
- Ka/Ks >1 alone is not a positive-selection significance test. Alignment quality, evolutionary distance and the scientific study design remain essential.

## Installation checks are a separate layer

Linux qualification covers native Ubuntu 22.04 CPU, Ubuntu 22.04 GPU container userspace on an Ubuntu 24.04 NVIDIA host, and a CentOS 7 CPU smoke test. Windows qualification covers CPU/GPU/reproduction, long and Unicode paths, compatible export and interrupted recovery. See [platform scope](LINUX_COMPATIBILITY.md), [Windows paths](WINDOWS_PATHS.md), [synthetic result observations](NUMERICAL_SMOKE.md) and [raw-CDS workflow checks](RAW_CDS_VALIDATION.md).

This documentation update does not refit data, change thresholds or claim new accuracy evidence. The distributed engines and released archive bytes are preserved.
