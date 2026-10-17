# Input and output contract

Inputs are directed codon-aligned pairs under genetic code 1. AXT or aligned FASTA with explicit pair_id/seq1/seq2 manifest is supported, including gzip. Reverse requests and requested order are preserved. Columns with a gap, ambiguous nucleotide or stop in either member are filtered jointly. Preparation metadata is byte/identity-checked. The main CLI does not align raw CDS or infer orthology. The optional [raw CDS/protein tutorial](RAW_CDS_TUTORIAL.md) ([中文](RAW_CDS_TUTORIAL_ZH.md)) supplies an external MAFFT/PAL2NAL batch workflow, followed by the unchanged prepare/validate commands.

The primary result provides pair identity, Ka, Ks and Ka/Ks, availability status and fitting diagnostics. Numeric missingness is retained as NA/null, not zero. Full models and statistics preserve all fourteen model slots. Quality summaries distinguish finite estimates, missing models, iteration caps, bounds, unknown gradients and gradients above the diagnostic threshold. The threshold is the infinity norm of the projected normalized log-gradient at 1e-8; it does not establish global optimality.

Run directories are new, never overwritten. Resume uses the original configuration and identity; relocation requires identical input bytes. Completed chunks are not equivalent to final summary commit. See the Chinese usage guide and generated examples for exact headers and supported commands.


1.1.0rc9 update: optional export

Completed standard results can be exported without fitting:
  kaksturbo export completed_run --format kaks-calculator --output new_export
The canonical result.tsv/results.jsonl remain unchanged. Missing statistics use NA with cell-level reasons. This is 22-column layout compatibility, not estimator equality. Full mappings, limits and reader examples: docs/COMPATIBLE_EXPORT.txt and docs/COMPATIBLE_EXPORT_ZH.txt. The Windows rc7 / Linux rc8 validation above is inherited only for byte-identical engines and runtimes; new rc9 export/installed-CLI checks are recorded separately in the new release QUALIFICATION.json. Download availability is listed in the current repository README.
