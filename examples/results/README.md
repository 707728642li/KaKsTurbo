# Synthetic result preview

These files are the unchanged output of the accepted **Windows 1.1.0rc10 CPU / robust** installation check on [`three_pair_smoke.axt`](../three_pair_smoke.axt). No numerical fit was performed to create the documentation preview.

- [`three_pair_cpu.tsv`](three_pair_cpu.tsv): full 18-column core result, three ordered requests.
- [`three_pair_cpu.jsonl`](three_pair_cpu.jsonl): the typed JSONL representation.
- [`three_pair_cpu_quality.txt`](three_pair_cpu_quality.txt): the actual diagnostic summary.
- [`provenance.json`](provenance.json): input/output hashes and version/scope.

There are two finite Ka/Ks ratios and one identical-sequence NA, retaining all 42 model slots in the source run. The README displays only selected columns rounded to six decimals. Empty sequence IDs reflect the absence of a metadata sidecar in this plain AXT fixture. Linux last digits and gradient flags can differ; [platform observations](../../docs/NUMERICAL_SMOKE.md) describe that distinction. This is synthetic installation data, not a biological selection study.
