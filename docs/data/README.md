# Aggregate timing data

These small tables support the software homepage and its reused Figure 2. They contain timings and run-level counts, not alignments or gene-level research output.

| File | Scope |
|:---|:---|
| `public1000_runs.tsv` | Nine same-host Windows observations: three arms × three technical runs |
| `cohort_cpu_runs.tsv` | Twenty archived Linux CPU observations: two cohorts × two arms × five runs |
| `scaling_runs.tsv` | Thirty archived observations: six physical-core allocations × five runs |
| `performance_summary.tsv` | Recomputed medians and minima/maxima of the first two tables |
| `scaling_summary.tsv` | Medians/ranges, speedups vs two cores and physical-core efficiency |

Values are derived without dropping any repeat. Internal historical arm label `official3` means the unmodified KaKs_Calculator 3.0 MA baseline; `cpu` / `gpu` mean KaKsTurbo. See [protocol and limitations](../PERFORMANCE.md). The older private 1,000-pair cohort and cross-platform historical GPU comparisons are not included in these tables.
