#!/usr/bin/env python3
"""External MAFFT/PAL2NAL recipe for frame-intact, code-1 CDS/PEP pairs.

No Ka/Ks fitting, homology inference, mismatch deletion, or frame repair.
Python standard library only; MAFFT and PAL2NAL are external executables.
GNU parallel is used for --jobs > 1, with one MAFFT thread per job.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

VERSION = "1.0"
BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = dict(zip((a+b+c for a in BASES for b in BASES for c in BASES), AA))
STOPS = {"TAA", "TAG", "TGA"}
PROTEIN = set("ACDEFGHIKLMNPQRSTVWY")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_fasta(path):
    path = Path(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    records = {}
    name, chunks = None, []
    def save():
        if name is not None:
            if not chunks or not "".join(chunks):
                raise ValueError(f"Empty FASTA sequence: {name}")
            records[name] = "".join(chunks)
    with opener(path, "rt", encoding="ascii") as stream:
        for number, text in enumerate(stream, 1):
            line = text.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                save()
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA ID at {path}:{number}")
                name = fields[0]
                if name in records:
                    raise ValueError(f"Duplicate FASTA ID: {name}; use unique transcript IDs")
                if any(ord(c) < 33 or ord(c) > 126 for c in name):
                    raise ValueError(f"Invalid FASTA ID: {name!r}")
                chunks = []
            elif name is None:
                raise ValueError(f"Sequence before FASTA header at {path}:{number}")
            elif any(c.isspace() for c in line):
                raise ValueError(f"Whitespace inside a sequence at {path}:{number}")
            else:
                chunks.append(line)
    save()
    if not records:
        raise ValueError(f"No FASTA records: {path}")
    return records


def read_pairs(path):
    with Path(path).open(encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["pair_id", "seq1", "seq2"]:
            raise ValueError("pairs.tsv needs exactly three tab-separated columns: pair_id, seq1, seq2")
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            if None in row or any(not row.get(k) for k in reader.fieldnames):
                raise ValueError(f"Missing or extra pair-table field on line {line}")
            if any(any(ord(c) < 32 or ord(c) > 126 for c in v) for v in row.values()):
                raise ValueError(f"Pair fields must be printable ASCII on line {line}")
            if row["pair_id"] in seen:
                raise ValueError(f"Duplicate pair_id on line {line}: {row['pair_id']}")
            seen.add(row["pair_id"])
            rows.append(row)
    if not rows:
        raise ValueError("No requested pairs")
    return rows


def check_cds_pep(name, cds, pep, strip_terminal_stop=False):
    if not cds or len(cds) % 3:
        raise ValueError(f"{name}: CDS length is not a positive multiple of 3")
    if set(cds.upper()) - set("ACGT"):
        raise ValueError(f"{name}: basic recipe requires unaligned A/C/G/T CDS; inspect ambiguous bases or gaps")
    has_stop, pep_stop = cds[-3:].upper() in STOPS, pep.endswith("*")
    if (has_stop or pep_stop) and not strip_terminal_stop:
        raise ValueError(f"{name}: terminal stop found; explicitly use --strip-terminal-stop to remove only that terminal codon/marker")
    if pep_stop and not has_stop:
        raise ValueError(f"{name}: peptide has a terminal * but CDS has no terminal stop; check annotation matching")
    removed = cds[-3:] if has_stop else ""
    dna = cds[:-3] if has_stop else cds
    protein = (pep[:-1] if pep_stop else pep).upper()
    if not dna or not protein or set(protein) - PROTEIN:
        raise ValueError(f"{name}: basic recipe requires the 20 standard amino acids, without internal stops, X or gaps")
    translated = "".join(CODE[dna[i:i+3].upper()] for i in range(0, len(dna), 3))
    if "*" in translated:
        raise ValueError(f"{name}: internal stop codon; inspect the CDS frame/annotation")
    if translated != protein:
        first = next((i+1 for i, (a, b) in enumerate(zip(translated, protein)) if a != b),
                     min(len(translated), len(protein)) + 1)
        raise ValueError(f"{name}: CDS translation != peptide under genetic code 1 (first difference at residue {first}; translated={len(translated)} aa, peptide={len(protein)} aa)")
    return {"cds": dna, "pep": protein, "removed_terminal_codon": removed,
            "removed_peptide_stop": pep_stop, "peptide_case_normalized": protein != pep.rstrip("*"),
            "original_cds_nt": len(cds), "retained_cds_nt": len(dna)}


def codon_projection(aligned_peptide, dna, protein):
    """Verify a PAL2NAL result against strict residue-to-original-codon mapping."""
    if aligned_peptide.replace("-", "").upper() != protein:
        raise ValueError("Protein alignment lost, added or changed residues")
    offset, codons = 0, []
    for amino_acid in aligned_peptide:
        if amino_acid == "-":
            codons.append("---")
        else:
            codons.append(dna[offset:offset+3])
            offset += 3
    if offset != len(dna):
        raise ValueError("Protein alignment does not consume the complete prepared CDS")
    return "".join(codons)


def pair_identity(pair_id, left, right):
    # Public input-metadata contract; historical separator is intentionally kept.
    digest = hashlib.sha256(b"KaKsForge-directed-aligned-pair-v1\0")
    for text in (pair_id, left, right):
        data = text.encode("ascii")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def write_fasta(path, records):
    Path(path).write_text("".join(f">{name}\n{seq}\n" for name, seq in records), encoding="ascii")


def run_tool(command, output, log, directory, timeout):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # MAFFT temp files stay in this new job directory, not in the source FASTAs.
    temporary = directory / "tmp"
    temporary.mkdir(exist_ok=True)
    env["TMPDIR"] = str(temporary)
    with output.open("w", encoding="ascii") as out, log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, stdout=out, stderr=err, cwd=directory, env=env,
                              timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed (exit {proc.returncode}); inspect {log.name}")


def worker(output, index):
    directory = Path(output).resolve() / ".work" / f"{int(index):08d}"
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    commands = []
    result = dict(pair_index=job["pair_index"], pair_id=job["pair_id"], seq1=job["seq1"], seq2=job["seq2"], complete=False)
    try:
        mafft = [job["tools"]["mafft"]["path"], "--amino", "--auto", "--thread", "1", "--inputorder", str(directory / "pep.fa")]
        commands.append(mafft)
        run_tool(mafft, directory / "pep.aln.fa", directory / "mafft.log", directory, job["timeout"])
        aligned = read_fasta(directory / "pep.aln.fa")
        if set(aligned) != {"S1", "S2"} or len({len(v) for v in aligned.values()}) != 1:
            raise ValueError("MAFFT must return exactly the two expected, equally long aligned proteins")
        expected = {key: codon_projection(aligned[key], job[key]["cds"], job[key]["pep"]) for key in ("S1", "S2")}
        pal = [job["tools"]["pal2nal"]["path"], str(directory / "pep.aln.fa"), str(directory / "cds.fa"), "-output", "fasta", "-codontable", "1"]
        commands.append(pal)
        run_tool(pal, directory / "codon.aln.fa", directory / "pal2nal.log", directory, job["timeout"])
        back = read_fasta(directory / "codon.aln.fa")
        if set(back) != {"S1", "S2"} or any(back[k].upper() != expected[k].upper() for k in expected):
            raise ValueError("PAL2NAL output is not an exact, full CDS back-translation; no trimming or mismatches accepted")
        left, right = expected["S1"], expected["S2"]
        gap_columns = sum("-" in left[i:i+3] + right[i:i+3] for i in range(0, len(left), 3))
        eligible = len(left) // 3 - gap_columns
        if not eligible:
            raise ValueError("No jointly eligible codons after alignment")
        matches = sum(a == b for a, b in zip(aligned["S1"].upper(), aligned["S2"].upper()) if a != "-" and b != "-")
        result.update(complete=True, left=left, right=right, protein_alignment=aligned,
                      aligned_codons=len(left)//3, gap_codons=gap_columns, eligible_codons=eligible,
                      peptide_identity_fraction=matches/eligible,
                      transformations={job["seq1"]: {k: v for k, v in job["S1"].items() if k not in ("cds", "pep")},
                                       job["seq2"]: {k: v for k, v in job["S2"].items() if k not in ("cds", "pep")}})
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    result.update(commands=commands, seconds=time.monotonic()-started)
    write_json(directory / "result.json", result)
    print(str(index) + "\t" + ("OK" if result["complete"] else "FAILED"), flush=True)
    return 0 if result["complete"] else 4


def tool_info(name, executable, arguments):
    path = shutil.which(executable)
    if not path:
        raise ValueError(f"Missing {name}: {executable}; install the optional raw-CDS environment first")
    path = str(Path(path).resolve())
    proc = subprocess.run([path, *arguments], capture_output=True, text=True, timeout=20, check=False)
    return {"path": path, "entry_file_sha256": sha(path), "version_or_help": (proc.stdout + proc.stderr).strip()[:4000]}


def errors_table(output, errors):
    with (output / "errors.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["pair_id", "seq1", "seq2", "error"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(errors)


def execute(args, output, manifest):
    paths = {k: Path(getattr(args, k)).expanduser().resolve() for k in ("cds", "pep", "pairs")}
    inputs = {k: {"path": str(v), "sha256": sha(v)} for k, v in paths.items()}
    manifest.update(inputs=inputs, script_sha256=sha(__file__), jobs=args.jobs, mafft_threads_per_job=1,
                    strip_terminal_stop=args.strip_terminal_stop, genetic_code=1)
    cds, pep, pairs = read_fasta(paths["cds"]), read_fasta(paths["pep"]), read_pairs(paths["pairs"])
    normalized, bad, errors = {}, {}, []
    for row in pairs:
        for name in (row["seq1"], row["seq2"]):
            if name in normalized or name in bad:
                continue
            try:
                if name not in cds or name not in pep:
                    raise ValueError(f"{name}: ID missing from CDS or peptide FASTA; names must match explicitly")
                normalized[name] = check_cds_pep(name, cds[name], pep[name], args.strip_terminal_stop)
            except ValueError as exc:
                bad[name] = str(exc)
        issues = [bad[name] for name in dict.fromkeys((row["seq1"], row["seq2"])) if name in bad]
        if issues:
            errors.append({**row, "error": "; ".join(issues)})
    manifest.update(requested_pairs=len(pairs), input_cds_records=len(cds), input_peptide_records=len(pep),
                    requested_unique_sequences=len(normalized)+len(bad))
    errors_table(output, errors)
    if errors:
        manifest.update(phase="input_failed", failed_pairs=len(errors), complete=False)
        raise ValueError(f"{len(errors)} requested pairs fail CDS/PEP validation; see errors.tsv. No alignment or final AXT written")
    tools = {"mafft": tool_info("MAFFT", args.mafft, ["--version"]),
             "pal2nal": tool_info("PAL2NAL", args.pal2nal, ["-h"])}
    if args.jobs > 1:
        if os.name == "nt":
            raise ValueError("Batch GNU parallel recipe is validated on Linux; prepare there and use its AXT on Windows")
        tools["parallel"] = tool_info("GNU parallel", args.parallel, ["--version"])
        if "GNU parallel" not in tools["parallel"]["version_or_help"]:
            raise ValueError("--parallel must select GNU parallel, not a different program named parallel")
    manifest.update(tools=tools, phase="aligning")
    write_json(output / "alignment_manifest.json", manifest)
    for index, row in enumerate(pairs):
        directory = output / ".work" / f"{index:08d}"
        directory.mkdir(parents=True)
        job = dict(row, pair_index=index, S1=normalized[row["seq1"]], S2=normalized[row["seq2"]], tools=tools, timeout=args.timeout)
        write_json(directory / "job.json", job)
        write_fasta(directory / "pep.fa", [("S1", job["S1"]["pep"]), ("S2", job["S2"]["pep"])])
        write_fasta(directory / "cds.fa", [("S1", job["S1"]["cds"]), ("S2", job["S2"]["cds"])])
    started, settled = time.monotonic(), 0
    def progress():
        elapsed = time.monotonic()-started
        rate = settled/max(elapsed, 1e-9)
        eta = (len(pairs)-settled)/rate if rate else 0
        print(f"Alignment [{settled}/{len(pairs)}] {rate:.2f} pairs/s | {elapsed:.1f}s elapsed | ETA {eta:.1f}s", file=sys.stderr, flush=True)
    if args.jobs == 1:
        for index in range(len(pairs)):
            worker(output, index)
            settled += 1
            progress()
    else:
        ids = output / "job_indices.txt"
        ids.write_text("".join(f"{i}\n" for i in range(len(pairs))), encoding="ascii")
        command = shlex.join([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_worker", str(output)]) + " {}"
        parallel = [tools["parallel"]["path"], "--plain", "--jobs", str(args.jobs), "--delay", "0.1",
                    "--line-buffer", "--joblog", str(output/"parallel.tsv"), command, "::::", str(ids)]
        manifest["parallel_command"] = parallel
        with (output/"parallel.stderr.log").open("w", encoding="utf-8") as err:
            proc = subprocess.Popen(parallel, stdout=subprocess.PIPE, stderr=err, text=True)
            for line in proc.stdout:
                if line.rstrip().endswith(("\tOK", "\tFAILED")):
                    settled += 1
                    progress()
            manifest["parallel_exit_code"] = proc.wait()
    records = []
    with gzip.open(output/"tool_logs.jsonl.gz", "wt", encoding="utf-8") as log:
        for index, row in enumerate(pairs):
            directory = output/".work"/f"{index:08d}"
            try:
                record = json.loads((directory/"result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = dict(row, pair_index=index, complete=False, error="Worker omitted its result; inspect parallel.stderr.log")
            records.append(record)
            log.write(json.dumps({"pair_index": index, "pair_id": row["pair_id"], "commands": record.get("commands", []),
                                  "logs": {p.name: p.read_text(encoding="utf-8", errors="replace") for p in directory.glob("*.log")}}) + "\n")
    errors = [{k: r[k] for k in ("pair_id", "seq1", "seq2", "error")} for r in records if not r["complete"]]
    errors_table(output, errors)
    fields = ["pair_index", "pair_id", "seq1", "seq2", "complete", "aligned_codons", "gap_codons", "eligible_codons", "peptide_identity_fraction"]
    with (output/"qc.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "NA") for k in fields})
    manifest.update(aligned_pairs=sum(r["complete"] for r in records), failed_pairs=len(errors))
    if errors or manifest.get("parallel_exit_code", 0):
        manifest["phase"] = "alignment_failed"
        raise ValueError(f"Alignment batch incomplete ({len(errors)} failed pairs); logs and work files retained, no final AXT")
    if any(sha(paths[k]) != inputs[k]["sha256"] for k in paths):
        raise ValueError("Original inputs changed during alignment; no final AXT committed")
    with (output/"aligned_pairs.axt").open("w", encoding="ascii", newline="\n") as axt, \
         (output/"aligned_pairs.metadata.jsonl").open("w", encoding="utf-8", newline="\n") as sidecar, \
         gzip.open(output/"protein_alignments.jsonl.gz", "wt", encoding="utf-8") as alignments, \
         (output/"pair_provenance.jsonl").open("w", encoding="utf-8", newline="\n") as provenance:
        for record in records:
            axt.write(f"{record['pair_id']}\n{record['left']}\n{record['right']}\n\n")
            sidecar.write(json.dumps(dict(pair_index=record["pair_index"], pair_id=record["pair_id"],
                stable_pair_sha256=pair_identity(record["pair_id"], record["left"], record["right"]),
                seq1_id=record["seq1"], seq2_id=record["seq2"],
                source_location={"format": "raw_cds_protein_guided", "path": inputs["pairs"]["path"], "line": record["pair_index"]+2})) + "\n")
            alignments.write(json.dumps({k: record[k] for k in ("pair_index", "pair_id", "seq1", "seq2", "protein_alignment")}) + "\n")
            provenance.write(json.dumps({k: v for k, v in record.items() if k not in ("left", "right", "protein_alignment")}) + "\n")
    # A complete manifest is committed last. Partial or interrupted batches are
    # never described as valid; users must check it before calling prepare/run.
    manifest.update(phase="complete", complete=True, source_unchanged=True,
                    codon_columns_removed=0, original_cds_case_preserved=True,
                    sha256={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "alignment_manifest.json"})
    if not args.keep_work:
        work = output/".work"
        if work.is_symlink() or work.resolve().parent != output.resolve():
            raise ValueError("Refuse cleanup outside the newly created output directory")
        shutil.rmtree(work)
    manifest["intermediates_retained"] = args.keep_work
    return manifest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return worker(argv[1], int(argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cds", "pep", "pairs", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent single-threaded pair alignments; Linux GNU parallel required above 1")
    parser.add_argument("--strip-terminal-stop", action="store_true", help="Explicitly remove one terminal CDS stop and corresponding optional PEP *; record every change")
    parser.add_argument("--keep-work", action="store_true", help="Keep successful per-pair intermediate files (failures always retained)")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds allowed per external tool per pair")
    parser.add_argument("--mafft", default="mafft")
    parser.add_argument("--pal2nal", default="pal2nal.pl")
    parser.add_argument("--parallel", default="parallel")
    args = parser.parse_args(argv)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if not 1 <= args.jobs <= available or args.timeout <= 0:
        parser.error(f"--jobs must be 1..{available} and --timeout positive")
    output = Path(args.output).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Error: choose a new writable output directory: {exc}", file=sys.stderr)
        return 2
    manifest = dict(schema="kaksturbo.raw_cds_alignment/1", recipe_version=VERSION, complete=False,
                    phase="preflight", fitting_performed=False, pairing_inferred=False)
    started = time.monotonic()
    code = 0
    try:
        execute(args, output, manifest)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        manifest.update(complete=False, error=str(exc))
        if not (output/"errors.tsv").exists():
            errors_table(output, [dict(pair_id="", seq1="", seq2="", error=str(exc))])
        print("Error: " + str(exc), file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        manifest.update(complete=False, phase="interrupted", error="Interrupted; use a new output directory after checking retained work")
        code = 130
    manifest["elapsed_seconds"] = time.monotonic()-started
    write_json(output/"alignment_manifest.json", manifest)
    if not code:
        print(f"Prepared {manifest['aligned_pairs']} directed pairs: {output/'aligned_pairs.axt'}\nNext: kaksturbo prepare --input <AXT> --metadata <sidecar> --output <new prepared directory>")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
