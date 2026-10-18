# Raw CDS and protein sequences to KaKsTurbo: a Linux quick start

[中文教程](RAW_CDS_TUTORIAL_ZH.md) · [Input/output contract](INPUT_OUTPUT.md)

```text
Raw CDS + matching proteins + explicit homologous-pair table
    → check CDS/protein agreement
    → MAFFT protein alignment for each pair
    → PAL2NAL codon back-translation
    → KaKsTurbo prepare / validate
    → KaKsTurbo run → estimates and quality summaries
```

KaKsTurbo's `prepare` command normalizes **already aligned** inputs. The separate tutorial script [`prepare_cds_pairs.py`](../scripts/prepare_cds_pairs.py) runs external alignment programs first; it does not modify the Ka/Ks estimator. Linux is the tested preparation platform. Prepared inputs can also be used on Windows.

## Inputs

Supply unaligned `cds.fa`, matching `pep.fa`, and a TAB-separated `pairs.tsv`:

```text
pair_id	seq1	seq2
pair_A_B	spA_tx1	spB_tx1
pair_B_A	spB_tx1	spA_tx1
pair_C_D	spA_tx2	spB_tx2
```

FASTA IDs are the first whitespace-delimited header field. IDs must be unique across the combined input and match between CDS and protein. Use the same annotation version and transcript isoform. Map protein/transcript identifiers explicitly when needed; names and file order are not a homology test. Unique pair IDs, directed requests and row order are retained, including reverse requests. A transcript may occur in several requested pairs.

Pairs must come from a justified homolog/ortholog/synteny analysis or a curated candidate list. The script neither infers orthology nor aligns unrelated gene families together. Choose transcript isoforms deliberately. This basic recipe requires coding-oriented, frame-intact CDS without introns/UTRs: A/C/G/T DNA and the 20 standard protein letters under **genetic code 1**. Ambiguity, internal stops, frameshifts and mismatches are reported for inspection, not repaired or silently discarded. The core program has separate rules for already aligned ambiguous input.

## Install the optional tools

From the extracted portable-package directory (or source directory), create a project-local environment; reuse an existing suitable one if available:

```bash
conda env create -p "$PWD/envs/input-prep" -f workflow/raw_cds_environment.yml
conda run -p "$PWD/envs/input-prep" --no-capture-output mafft --version
conda run -p "$PWD/envs/input-prep" --no-capture-output pal2nal.pl -h
conda run -p "$PWD/envs/input-prep" --no-capture-output parallel --version
```

Tested tools: MAFFT 7.526, Bioconda PAL2NAL 14.1, and GNU parallel 20260722. Upstream PAL2NAL now also provides a newer implementation; this recipe pins the tested Perl interface and does not claim identical behavior for other versions. These tools are external prerequisites for alignment, not additions to the KaKsTurbo binary runtime.

## Run the four-request demo

The [synthetic demo](../examples/raw_cds_demo) covers two gene families, a reverse request and an identical-sequence control. It illustrates formats and execution, not biological accuracy or benchmark performance. The commands below run from the extracted Linux package directory, so `KT="$PWD/kaksturbo"` selects the bundled executable. In the source tree, replace it with your installed executable's absolute path. Every output directory must be new. The [quick start](../QUICKSTART.md) also provides a one-command demo runner.

```bash
set -euo pipefail
PREP_ENV="$PWD/envs/input-prep"
KT="$PWD/kaksturbo"

conda run -p "$PREP_ENV" --no-capture-output python scripts/prepare_cds_pairs.py \
  --cds examples/raw_cds_demo/cds.fa --pep examples/raw_cds_demo/pep.fa \
  --pairs examples/raw_cds_demo/pairs.tsv \
  --output tutorial_alignment --jobs 2 --strip-terminal-stop

"$KT" prepare --input tutorial_alignment/aligned_pairs.axt \
  --metadata tutorial_alignment/aligned_pairs.metadata.jsonl --output tutorial_prepared
"$KT" validate --input tutorial_prepared
"$KT" run --input tutorial_prepared --output tutorial_results --backend cpu --threads 2
```

Only proceed when preparation exits zero and `alignment_manifest.json` says `complete: true`. The explicit `--strip-terminal-stop` option removes at most one terminal CDS stop and its optional protein `*`, recording each transformation. Internal stops and translation mismatches still fail. Omit the flag when no terminal stop needs removal. Original files and retained CDS letter case are preserved.

The actual Linux demo produced four result rows: three finite Ka/Ks estimates and one NA for the identical-sequence control. NA remains missing, never zero. This small exercise makes no general numerical-convergence or timing claim.

## Use your own data in parallel

Replace the three demo paths. For example:

```bash
set -euo pipefail
PREP_ENV="$PWD/envs/input-prep"
KT="$PWD/kaksturbo"

conda run -p "$PREP_ENV" --no-capture-output python scripts/prepare_cds_pairs.py \
  --cds data/cds.fa --pep data/pep.fa --pairs data/pairs.tsv \
  --output alignment_batch --jobs 8 --strip-terminal-stop
"$KT" prepare --input alignment_batch/aligned_pairs.axt \
  --metadata alignment_batch/aligned_pairs.metadata.jsonl --output prepared_batch
"$KT" validate --input prepared_batch
"$KT" run --input prepared_batch --output results_cpu --backend cpu --threads 8
```

`--jobs 8` permits eight independent pair alignments, each with one MAFFT thread. GNU parallel staggers launches. Alignment and KaKsTurbo fitting run sequentially; their concurrency settings describe different stages. Progress reports actual settled requests, throughput, elapsed time and ETA. `--jobs 1` needs no GNU parallel. `--timeout 300` limits each external alignment command per pair, not the model-fitting iteration count.

On success, compact logs and protein alignments are retained and duplicate work files are removed. `--keep-work` retains them for inspection. On failure, work files and an error table remain; **no final subset AXT is exported**. Fix the inputs and use a new output directory.

For GPU fitting, replace the final `run` with:

```bash
nvidia-smi
"$KT" doctor --backend gpu --probe-gpu
"$KT" run --input prepared_batch --output results_gpu --backend gpu --device cuda:0
```

Each fit uses one GPU. The precompiled package needs a compatible NVIDIA driver, not a separately installed CUDA Toolkit. Explicit GPU requests report availability failures; CPU execution remains available. Consult the installation guide for device requirements. The new tutorial's end-to-end execution test used Linux CPU; it does not claim a new GPU qualification.

## Alignment details and output

Per pair, temporary `S1/S2` aliases protect original identifiers from external-tool naming conventions. CDS and PEP use identical names and order. The actual commands are:

```bash
mafft --amino --auto --thread 1 --inputorder pair.pep.fa > pair.pep.aln.fa
pal2nal.pl pair.pep.aln.fa pair.cds.fa -output fasta -codontable 1 > pair.codon.aln.fa
```

The helper verifies complete CDS/protein translation, preservation of every protein residue, and exact back-translation of all retained CDS triplets. It does not use PAL2NAL's `-nogap` or `-nomismatch`. Triplet gaps remain for the core program's established paired-codon filter. It never selects requests by alignment score or final Ka/Ks.

Pairwise alignment is appropriate for independent pairwise requests. For a study requiring common coordinates across several species, first construct an alignment within each homologous group and extract the requested pairs from that shared codon alignment. Do not treat these two designs as interchangeable.

| Output | Meaning |
|---|---|
| `aligned_pairs.axt` | KaKs three-line AXT records, with blank separators; not genomic/UCSC AXT |
| `aligned_pairs.metadata.jsonl` | Original sequence IDs, direction, order and exact sequence fingerprints |
| `alignment_manifest.json` | Completion, original input hashes, tool identities and policy |
| `qc.tsv` / `errors.tsv` | Coverage, gap codons, eligible codons, protein identity and explicit failures |
| `pair_provenance.jsonl` | Commands, preparation time and terminal-stop transformations |
| `protein_alignments.jsonl.gz` / `tool_logs.jsonl.gz` | Compressed alignments and logs for review |
| `parallel.tsv` | Batch job log when multiple workers are used |

An AXT record looks like:

```text
pair_A_B
ATGGCT---TTT
ATGGCCGGATTC

```

Use the metadata together with AXT. `kaksturbo prepare` then creates a standard directory with bound hashes; preserve or move the whole directory. Inspect `result.tsv`, `results.jsonl` and `quality_summary.txt` after fitting. Finite Ka/Ks is not a convergence certificate, and a ratio above one is not by itself a significance test. Inspect questionable alignments; the recipe does not silently remove low-identity pairs.

## Common failures

| Failure | Check |
|---|---|
| Missing/duplicate ID | Globally unique IDs, transcript mapping and matching annotation versions |
| CDS translation differs from peptide | Frame, strand, isoform and unintended UTR/intron sequence |
| Internal stop or non-triplet CDS length | Annotation/extraction errors; do not delete bases merely to pass |
| Terminal stop | Use the explicit stripping option after confirming a legitimate terminal stop |
| Ambiguity or `X` | Inspect the annotation; this basic workflow does not repair or mask it |
| Tool error or empty output | Logs and retained `.work/`; exit zero alone never passes acceptance |
| Metadata/hash failure | Keep the prepared files together and avoid modifying one in isolation |

After Linux preparation, copy the complete prepared directory to Windows if needed. Windows rc10 handles long input/output paths, spaces and Unicode without changing the system long-path policy. See WINDOWS_PATHS.md for deep-installation launch and safe extraction. The scientific engine is unchanged from rc9. Adjust the example drive and use a new directory:

```powershell
.\kaksturbo.exe validate --input prepared_batch
.\kaksturbo.exe run --input prepared_batch --output D:/kt_run1 --backend cpu --threads 8
```

## Sources and scope

- [MAFFT](https://mafft.cbrc.jp/alignment/software/) and its [threading notes](https://mafft.cbrc.jp/alignment/software/multithreading.html).
- [PAL2NAL](https://www.bork.embl.de/pal2nal/), [Suyama et al. (2006)](https://doi.org/10.1093/nar/gkl315). The strict whole-CDS checks are additional checks in this helper.
- [GNU parallel](https://www.gnu.org/software/parallel/); follow its citation guidance when used in published work.

The preparation helper preserves the scientific engine, fourteen models, FP64 and numerical gates. Its published [workflow validation record](RAW_CDS_VALIDATION.md) and [machine-readable summary](RAW_CDS_VALIDATION.json) describe the tested inputs and platform. These checks qualify the tutorial; they are not performance or biological-accuracy evidence.

See the [qualification record](RAW_CDS_VALIDATION.md) for exact platform, checks and limits.
