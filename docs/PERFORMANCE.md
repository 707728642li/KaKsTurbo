# Performance evidence

[Home](../README.md#performance-and-scaling) · [Underlying observations](data/) · [Methods](METHODS.md)

Performance is reported through three complementary experiments: a same-host comparison, complete plant/fly cohorts, and physical-core scaling. Each uses its stated input, frozen software build, and execution protocol. All technical repeats are retained, and timing covers the complete CLI workflow. Results describe these workloads and conditions.

## A. Public 1,000-pair panel: same-host comparison

All arms ran on the same native Windows workstation and storage, with **eight physical host cores / sixteen logical processors**. The KaKs_Calculator 3.0 MA baseline used **sixteen independent CLI workers** over 64 length-balanced chunks; this is not a claim of built-in sixteen-thread execution. KaKsTurbo GPU used a single RTX 3090 with the same host-core budget. Its CPU and GPU implementations retain the same fourteen-model, FP64, bounded-optimization contract.

| Arm | N | Median (s) | Minimum–maximum (s) | Baseline / median |
|:---|---:|---:|:---|---:|
| KaKs_Calculator 3.0 MA | 3 | 458.044111 | 455.507405–458.472112 | 1.00× |
| KaKsTurbo CPU | 3 | 22.442249 | 22.424163–22.710194 | 20.41× |
| KaKsTurbo GPU | 3 | 23.059449 | 22.939519–23.218917 | 19.86× |

The GPU is slightly slower than the CPU on this workload. All nine runs retain 1,000 requested pairs and 14,000 model slots, with 1,000 finite MA ratios and no MA NA. Saved-endpoint gradient flags remain: 353 CPU model slots and 254 GPU model slots are above the strict 1e-8 diagnostic threshold in each repeat. The baseline does not export comparable gradient diagnostics.

The balanced panel contains 250 requests from each of four public plant/animal contrasts. Selection used sequence characteristics and hash priority, not measured speed or final Ka/Ks. This panel is **different from the older private 1,000-pair benchmark**; the older numbers are not substituted here.

- Input SHA256: `02f5ba0d1439d20470a2aa75749672611a47298b149969ab90653d5f4ab5dc16`.
- Timing: fresh process launch through completed standard output, including baseline merging and KaKsTurbo output/QC. Independent correctness follow-up and compression are outside the timer.
- All three technical runs and their predetermined order are retained. Own local numerical tasks were serial; no controlled cold-disk-cache claim is made.
- [All nine timing and coverage observations](data/public1000_runs.tsv).

## B. Public plant and fly cohorts: archived CPU comparisons

These use **bioserver Linux**, with a matched eight-physical-core / sixteen-logical-thread allocation per arm. They are archived frozen benchmark builds, kept separate from panel A. Jobs ran in a concurrent batch with disjoint CPU allocations; shared memory, storage and host activity can still affect wall time.

| Cohort | N per arm | Baseline median (s) | CPU median (s) | Baseline / CPU |
|:---|---:|---:|---:|---:|
| Plants, 29,028 directed pairs | 5 | 17,385.788798 | 1,520.302752 | 11.44× |
| Flies, 23,571 directed pairs | 5 | 55,989.671109 | 1,221.293547 | 45.84× |

The plant cohort combines *Arabidopsis thaliana–A. lyrata* and *A. thaliana–Arabis alpina* comparisons. The fly cohort combines *Drosophila melanogaster–D. simulans* and *D. melanogaster–D. yakuba*. They are cross-species pair collections, not a single-species Arabidopsis experiment. Sequence composition, length and optimizer workload differ; the ratios need not match across cohorts.

Only the same-platform CPU comparisons from Figure 2B are summarized here. Historical GPU cohort observations used a different Windows host and mixed serial/concurrent execution; they are not folded into a same-hardware acceleration claim.

[All twenty CPU observations](data/cohort_cpu_runs.tsv) · [Unrounded medians and ranges](data/performance_summary.tsv)

## C. Physical-core scaling: a separate fixed 10,000-pair panel

This archived CPU experiment uses one hardware thread per physical core on the same bioserver host. Each core allocation has **five technical runs**. Speedup is relative to the **two-core KaKsTurbo median**, not relative to KaKs_Calculator.

| Physical cores | Median complete CLI (s) | Speedup vs 2 cores | Parallel efficiency |
|---:|---:|---:|---:|
| 2 | 2,171.00 | 1.00× | 100.0% |
| 4 | 1,060.46 | 2.05× | 102.4% |
| 8 | 577.01 | 3.76× | 94.1% |
| 16 | 370.85 | 5.85× | 73.2% |
| 32 | 179.89 | 12.07× | 75.4% |
| 64 | 98.13 | 22.12× | 69.1% |

Efficiency = speedup / (physical cores / 2). The four-core value slightly above 100% is an observed timing ratio, not evidence of universally superlinear scaling; cache and shared-host variation remain possible contributors. The 64-core result is sublinear. **64 physical cores and 128 SMT threads are not interchangeable.**

[All thirty observations](data/scaling_runs.tsv) · [Unrounded scaling statistics](data/scaling_summary.tsv)

## Interpretation and reproducibility

All ratios divide **unrounded medians**. Error bars are observed minima and maxima, not confidence intervals. N counts technical runs, not independent biological samples. No slow repeat is removed.

These are complete-workflow comparisons: robust-mode numerical policies differ from the baseline, so the speed ratios do not isolate hardware acceleration or prove identical statistical estimates. The historical and same-host panels are not pooled. Figure 2 contains N=3 and N=5 panels deliberately; it must not be described globally as N=5.

The published tables contain aggregate timings and run-level coverage only; no private alignments, gene-level research results or raw benchmark intermediates are included. [Figure provenance](assets/figure_provenance.json) records the unchanged image hashes. [Numerical validation](NUMERICAL_VALIDATION.md) discusses accuracy and unresolved cases separately.
