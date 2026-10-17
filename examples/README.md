# Small examples

`aligned/` contains CLI-generated aligned FASTA, pair manifest and AXT examples. `three_pair_smoke.axt` is deterministic synthetic sequence data: a variable pair, its reverse, and an identical pair. There are 122 sense codons per sequence. It is installation test data, not a biological selection benchmark.

Run `kaksturbo run --input examples/three_pair_smoke.axt --output smoke --backend cpu --threads 2`. Expect three ordered requests, all 42 model slots, two finite MA ratios near 1.00615 and one explicit NA for the identical pair. The public quality summary must preserve endpoint warnings. Windows/Linux last digits and projected-gradient flags can differ; do not turn these observations into a convergence guarantee.
