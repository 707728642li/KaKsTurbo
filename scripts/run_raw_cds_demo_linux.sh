#!/usr/bin/env bash
# Run the bundled synthetic example with optional upstream tools, then the
# unchanged installed CPU engine. Alignment and fitting execute sequentially.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
prep_env="${root_dir}/envs/input-prep"
output_dir="${PWD}/tutorial_demo"
jobs=2
threads=2
conda_bin=conda
parallel_bin=parallel
kt_bin="${KAKSTURBO_BIN:-${root_dir}/kaksturbo}"

usage() {
    cat <<'HELP'
Run the bundled raw-CDS demo on Linux (no GPU required).
  bash scripts/run_raw_cds_demo_linux.sh [options]
  --env PATH        Existing preparation Conda environment (default: ./envs/input-prep in package)
  --output PATH     New output directory; parent must exist (default: ./tutorial_demo)
  --jobs N          Concurrent single-threaded alignments (default: 2)
  --threads N       CPU fitting threads, after alignment (default: 2)
  --conda PATH      Conda executable (default: conda)
  --parallel PATH   Existing GNU parallel executable (default: parallel)
  -h, --help        Show this help
Set KAKSTURBO_BIN to an installed absolute executable path when using the source tree.
Install or reuse MAFFT/PAL2NAL/Python and, for jobs > 1, GNU parallel; see QUICKSTART.md.
This script does not install tools, infer homology, overwrite outputs, or change fitting settings.
HELP
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --env|--output|--jobs|--threads|--conda|--parallel)
            if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
            case "$1" in
                --env) prep_env="$2" ;;
                --output) output_dir="$2" ;;
                --jobs) jobs="$2" ;;
                --threads) threads="$2" ;;
                --conda) conda_bin="$2" ;;
                --parallel) parallel_bin="$2" ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ || ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Jobs and threads must be positive integers.\n' >&2; exit 2
fi
if [[ ! -d "$prep_env" ]]; then
    printf 'Preparation environment missing: %s\nSee QUICKSTART.md, or use --env with an existing environment.\n' "$prep_env" >&2
    exit 2
fi
if ! command -v "$conda_bin" >/dev/null 2>&1; then
    printf 'Conda not found: %s\nUse --conda /absolute/path/to/conda.\n' "$conda_bin" >&2; exit 2
fi
if [[ ! -x "$kt_bin" ]]; then
    printf 'Linux executable missing or not executable: %s\nExtract the full Linux portable package, or set KAKSTURBO_BIN.\n' "$kt_bin" >&2
    exit 2
fi
prep_env="$(cd -- "$prep_env" && pwd)"
# mkdir (without -p) rejects existing outputs, preserving every previous run.
if ! mkdir -- "$output_dir"; then
    printf 'Choose a new output directory with an existing parent. No previous output was changed.\n' >&2
    exit 2
fi
output_dir="$(cd -- "$output_dir" && pwd)"
printf '[1/4] Checking CPU availability\n' >&2
"$kt_bin" doctor --backend cpu
printf '[2/4] Aligning the four synthetic directed requests\n' >&2
"$conda_bin" run -p "$prep_env" --no-capture-output python -I -B \
    "$root_dir/scripts/prepare_cds_pairs.py" \
    --cds "$root_dir/examples/raw_cds_demo/cds.fa" \
    --pep "$root_dir/examples/raw_cds_demo/pep.fa" \
    --pairs "$root_dir/examples/raw_cds_demo/pairs.tsv" \
    --output "$output_dir/alignment" --jobs "$jobs" --parallel "$parallel_bin" --strip-terminal-stop
printf '[3/4] Binding and validating the codon-aligned input\n' >&2
"$kt_bin" prepare --input "$output_dir/alignment/aligned_pairs.axt" \
    --metadata "$output_dir/alignment/aligned_pairs.metadata.jsonl" --output "$output_dir/prepared"
"$kt_bin" validate --input "$output_dir/prepared"
printf '[4/4] Running the unchanged fourteen-model CPU calculation\n' >&2
"$kt_bin" run --input "$output_dir/prepared" --output "$output_dir/results_cpu" --backend cpu --threads "$threads"
cat -- "$output_dir/results_cpu/quality_summary.txt"
printf '\nDemo completed. Main result: %s\nThis synthetic workflow check is not a biological or performance benchmark.\n' "$output_dir/results_cpu/result.tsv"
