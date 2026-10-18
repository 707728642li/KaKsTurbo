# Native installation observations

Both existing-host clean-target installations passed ten CLI checks on Python 3.11. The three-pair input produced 42 model records, two finite MA results and one identical-pair NA. Completed-run resume preserved every final artifact hash. This is installation coverage, not an independent convergence study or a clean-OS portability certificate.

Windows: Ka/Ks 1.0061535222983835 for the variable pair and reverse; 28 finite-model gradients within 1e-8.
Linux: Ka/Ks 1.0061535285550665 for the same pair and reverse; 26 within and 2 above 1e-8. Fourteen model diagnostics associated with the NA pair remain unknown on both platforms. The difference in final Ka/Ks is about 6.22e-9 relative; no finite/NA or above/below-one change occurs. The Linux warnings are retained, not suppressed or reclassified as converged.

The [exact synthetic input](../examples/three_pair_smoke.axt) and [Windows output preview](../examples/results/README.md) are included; the platform observations above retain their distinct diagnostic counts. The package's output schema and quality summaries expose these distinctions directly.
