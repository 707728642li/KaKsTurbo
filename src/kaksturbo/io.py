"""Streaming code-1 aligned input, with disk-backed duplicate/FASTA indexes."""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from ._paths import absolute_path
import sqlite3
import tempfile
from typing import Iterable, Iterator
from .errors import ConfigurationError, InputError
from .models import AlignedPair, RunConfig

ALPHABET = frozenset("ACGTRYSWKMBDHVNacgtryswkmbdhvn-?")
STOPS = frozenset(("TAA", "TAG", "TGA"))


def open_text(path: Path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    return opener(path, "rt", encoding="ascii", newline=None)


def pair_record(pair: AlignedPair) -> bytes:
    return (pair.pair_id + "\n" + pair.left + "\n" + pair.right + "\n\n").encode("ascii")


def pair_identity(pair: AlignedPair) -> str:
    # Length-prefixed fields avoid ambiguity and preserve direction/case/gaps.
    # Historical domain separator preserves all existing pair identifiers.
    digest = hashlib.sha256(b"KaKsForge-directed-aligned-pair-v1\0")
    for value in (pair.pair_id, pair.left, pair.right):
        data = value.encode("ascii")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def check_pair(pair: AlignedPair) -> dict:
    name, left, right = pair.pair_id, pair.left, pair.right
    if not isinstance(name, str) or not name or any(ord(c) < 32 or ord(c) > 126 for c in name):
        raise InputError("Pair IDs must be nonempty printable ASCII without tabs or newlines")
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        raise InputError(f"Empty aligned sequence: {name}")
    if len(left) != len(right):
        raise InputError(f"Unequal aligned sequence lengths: {name}", details={"left": len(left), "right": len(right)})
    if len(left) % 3:
        raise InputError(f"Aligned sequence length is not divisible by three: {name}")
    invalid = set(left + right) - ALPHABET
    if invalid:
        raise InputError(f"Invalid DNA alignment symbols: {name}", details={"symbols": sorted(invalid)})
    # The historical filter uppercases codons, while original input case still affects raw GC.
    clean = ambiguity = gaps = stops = 0
    for position in range(0, len(left), 3):
        a, b = left[position:position + 3].upper(), right[position:position + 3].upper()
        if "-" in a + b:
            gaps += 1
        if set(a + b) - set("ACGT"):
            ambiguity += 1
            continue
        if a in STOPS or b in STOPS:
            stops += 1
            continue
        clean += 1
    return {"aligned_nt": len(left), "aligned_codons": len(left) // 3,
            "eligible_codons_preflight": clean, "ambiguous_codons": ambiguity,
            "gap_codons": gaps, "stop_codons": stops,
            "status": "no_eligible_codons" if not clean else "valid",
            "filter_policy": "historical paired ACGT/nonstop codon filter after case folding; raw bytes retained for GC"}


def _location(path, kind, record=None, line=None, line_end=None):
    result = {"path": str(absolute_path(path)), "format": kind}
    for key, value in (("record", record), ("line", line), ("line_end", line_end)):
        if value is not None:
            result[key] = value
    return result


def _located(error, location, **details):
    values = {**error.details, **details, "source_location": location}
    where = location["path"]
    if "line" in location:
        where += f":{location['line']}"
    if "record" in location:
        where += f" ({location['format']} record {location['record']})"
    return InputError(f"{error} [{where}]", details=values)


def _raise(error):
    raise error


def infer_input_format(input_path) -> str:
    """Sniff a raw first nonblank line, leaving all text errors to validation.

    ``>`` indicates FASTA; everything else (including an empty input) is AXT.
    The CLI can override this heuristic for an AXT pair ID beginning with ``>``.
    """
    path = absolute_path(input_path)
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    try:
        with opener(path, "rb") as handle:
            for raw in handle:
                # Also recognize the universal-newline inputs accepted by the
                # strict readers without decoding later invalid ASCII bytes.
                for line in raw.splitlines():
                    if line.strip():
                        return "fasta" if line.lstrip().startswith(b">") else "axt"
    except (OSError, EOFError) as exc:
        raise _located(InputError(f"Cannot read input for format detection: {exc}"),
                       _location(path, "auto")) from exc
    return "axt"


def _source_lines(path, kind, report_error):
    """Read universal-newline ASCII with exact physical line numbers, also for gzip."""
    path = Path(path)
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    try:
        # Surrogate escape avoids TextIO readahead moving a decoding error to an
        # earlier line. Invalid bytes are reported and never passed to a parser.
        with opener(path, "rb") as raw:
            with io.TextIOWrapper(raw, encoding="ascii", errors="surrogateescape", newline=None) as handle:
                for number, text in enumerate(handle, 1):
                    text = text.rstrip("\r\n")
                    invalid = next((i for i, c in enumerate(text) if ord(c) > 127), None)
                    if invalid is not None:
                        report_error(_located(InputError("Input must contain ASCII text"),
                                              _location(path, kind, line=number), column=invalid + 1))
                        yield number, None
                    else:
                        yield number, text
    except (UnicodeError, OSError, EOFError) as exc:
        report_error(_located(InputError(f"Cannot read input: {exc}"), _location(path, kind)))


def _iter_axt_records(path, report_error=_raise, counters=None):
    block, numbers, damaged, record = [], [], False, 0

    def finish():
        nonlocal record
        record += 1
        if counters is not None:
            counters["pairs_seen"] += 1
        location = _location(path, "axt", record, numbers[0], numbers[-1])
        if damaged:
            return None
        try:
            pair, quality = _axt_block(block, with_quality=True)
        except InputError as exc:
            # Point invalid symbols at the exact physical source line; other
            # pair-wide errors point at the record header and give its extent.
            for number, line in zip(numbers[1:], block[1:]):
                invalid = next((i for i, char in enumerate(line) if char not in ALPHABET), None)
                if invalid is not None:
                    location = {**location, "line": number}
                    report_error(_located(exc, location, pair_id=block[0], column=invalid + 1))
                    return None
            report_error(_located(exc, location, pair_id=block[0] if block else None))
            return None
        return pair, {"seq1_id": None, "seq2_id": None, "source_location": location, "_qc": quality}

    for number, line in _source_lines(path, "axt", report_error):
        if line is not None and not line.strip():
            if numbers:
                result = finish()
                if result is not None:
                    yield result
                block, numbers, damaged = [], [], False
        else:
            numbers.append(number)
            if line is None:
                damaged = True
            else:
                block.append(line)
    if numbers:
        result = finish()
        if result is not None:
            yield result


def iter_axt(path: Path) -> Iterator[AlignedPair]:
    for pair, _ in _iter_axt_records(path):
        yield pair


def _axt_block(lines: list[str], *, with_quality=False):
    if len(lines) < 3:
        raise InputError(f"AXT record needs a pair ID and two aligned sequences: {lines[0]!r}")
    if len(lines) == 3 and len(lines[1]) != len(lines[2]):
        raise InputError(f"Unequal two-line AXT sequences: {lines[0]}")
    # Original wrapped AXT concatenates the body and splits it at the byte midpoint.
    body = "".join(lines[1:])
    if len(body) % 6:
        raise InputError(f"AXT body cannot form two equally long codon alignments: {lines[0]}")
    middle = len(body) // 2
    pair = AlignedPair(lines[0], body[:middle], body[middle:])
    quality = check_pair(pair)
    return (pair, quality) if with_quality else pair


def _iter_fasta_records(path, manifest, index_path, report_error=_raise, counters=None):
    database = sqlite3.connect(index_path)
    try:
        database.execute("CREATE TABLE sequences (id TEXT PRIMARY KEY, sequence TEXT NOT NULL, location TEXT, invalid INTEGER)")
        current, sequence, location, invalid = None, [], None, False
        record = 0

        def save():
            if current is None:
                return
            joined = "".join(sequence)
            bad = invalid
            if not joined:
                report_error(_located(InputError(f"Empty FASTA sequence: {current}"), location, sequence_id=current))
                bad = True
            elif len(joined) % 3:
                report_error(_located(InputError(f"Aligned FASTA length is not divisible by three: {current}"),
                                      location, sequence_id=current, aligned_nt=len(joined)))
                bad = True
            try:
                database.execute("INSERT INTO sequences VALUES (?, ?, ?, ?)",
                                 (current, joined, json.dumps(location), int(bad)))
            except sqlite3.IntegrityError:
                first = database.execute("SELECT location FROM sequences WHERE id=?", (current,)).fetchone()
                # A duplicate ID is ambiguous; never select its first sequence.
                database.execute("UPDATE sequences SET invalid=1 WHERE id=?", (current,))
                report_error(_located(InputError(f"Duplicate FASTA sequence ID: {current}"), location,
                                      sequence_id=current, first_location=json.loads(first[0])))

        for number, line in _source_lines(path, "fasta", report_error):
            if line is None:
                invalid = True
                continue
            if line.startswith(">"):
                save()
                record += 1
                location = _location(path, "fasta", record, number, number)
                fields = line[1:].split()
                current, sequence, invalid = None, [], False
                if not fields or any(ord(c) < 33 or ord(c) > 126 for c in fields[0]):
                    report_error(_located(InputError("FASTA sequence ID must be nonempty printable ASCII"), location))
                    continue
                current = fields[0]
            elif line.strip():
                if current is None:
                    report_error(_located(InputError("FASTA sequence data precedes a valid header"),
                                          _location(path, "fasta", record or None, number, number)))
                    continue
                location["line_end"] = number
                sequence.append(line)
                column = next((i for i, char in enumerate(line) if char not in ALPHABET), None)
                if column is not None:
                    invalid = True
                    report_error(_located(InputError(f"Invalid DNA alignment symbols: {current}"),
                                          {**location, "line": number}, sequence_id=current,
                                          column=column + 1, symbols=sorted(set(line) - ALPHABET)))
        save()
        database.commit()
        header_seen, header_valid, row_number = False, False, 0
        for number, line in _source_lines(manifest, "pairs_tsv", report_error):
            if line is not None and not line and header_seen:
                continue
            if not header_seen:
                header_seen = True
                location = _location(manifest, "pairs_tsv", line=number, line_end=number)
            else:
                row_number += 1
                if counters is not None:
                    counters["pairs_seen"] += 1
                location = _location(manifest, "pairs_tsv", row_number, number, number)
            if line is None:
                continue
            try:
                values = next(csv.reader([line], delimiter="\t", strict=True))
            except csv.Error as exc:
                report_error(_located(InputError(f"Malformed pairs TSV row: {exc}"), location))
                continue
            if row_number == 0:
                header_valid = values in (["pair_id", "seq1", "seq2"], ["id", "seq1", "seq2"])
                if not header_valid:
                    report_error(_located(InputError("Pairs TSV requires exactly pair_id (or id), seq1, seq2 columns in this order"), location))
                continue
            if len(values) != 3:
                report_error(_located(InputError("Malformed pairs TSV row: expected three tab-separated fields"), location))
                continue
            if not header_valid:
                continue
            name, left_id, right_id = values
            sequences, locations, bad = [], [], False
            for key, sequence_id in (("seq1", left_id), ("seq2", right_id)):
                found = database.execute("SELECT sequence, location, invalid FROM sequences WHERE id=?", (sequence_id,)).fetchone()
                if found is None or found[2]:
                    problem = "Missing" if found is None else "Invalid or duplicate"
                    report_error(_located(InputError(f"{problem} FASTA sequence ID in pair manifest: {sequence_id}"),
                                          location, pair_id=name, field=key, sequence_id=sequence_id))
                    bad = True
                else:
                    sequences.append(found[0])
                    locations.append(json.loads(found[1]))
            if bad:
                continue
            pair = AlignedPair(name, *sequences)
            metadata = {"seq1_id": left_id, "seq2_id": right_id, "source_location": location,
                        "seq1_location": locations[0], "seq2_location": locations[1]}
            try:
                quality = check_pair(pair)
            except InputError as exc:
                report_error(_located(exc, location, pair_id=name, seq1_location=locations[0], seq2_location=locations[1]))
                continue
            yield pair, {**metadata, "_qc": quality}
        if not header_seen:
            report_error(_located(InputError("Empty pairs TSV: missing header"), _location(manifest, "pairs_tsv", line=1)))
    finally:
        database.close()


def iter_fasta_pairs(path: Path, manifest: Path, index_path: Path) -> Iterator[AlignedPair]:
    """Index aligned FASTA on disk; explicit TSV rows determine order and direction."""
    for pair, _ in _iter_fasta_records(path, manifest, index_path):
        yield pair


def input_pair_records(path, config, workspace, *, report_error=_raise, counters=None):
    """Validated directed pairs plus source metadata; duplicate IDs are disk-backed."""
    database = sqlite3.connect(Path(workspace) / "pair_ids.sqlite")
    database.execute("CREATE TABLE pair_requests (id TEXT PRIMARY KEY, location TEXT NOT NULL)")
    try:
        records = (_iter_axt_records(path, report_error, counters) if config.input_format == "axt" else
                   _iter_fasta_records(path, Path(config.pairs_manifest), Path(workspace) / "fasta.sqlite", report_error, counters))
        from .input_metadata import attach_metadata
        records = attach_metadata(records, config.input_metadata, report_error)
        for pair, metadata in records:
            location = metadata["source_location"]
            try:
                database.execute("INSERT INTO pair_requests VALUES (?, ?)", (pair.pair_id, json.dumps(location)))
            except sqlite3.IntegrityError:
                first = database.execute("SELECT location FROM pair_requests WHERE id=?", (pair.pair_id,)).fetchone()
                report_error(_located(InputError(f"Duplicate pair ID across the complete input: {pair.pair_id}"),
                                      location, pair_id=pair.pair_id, first_location=json.loads(first[0])))
                continue
            quality = metadata["_qc"]
            yield pair, {**metadata, "aligned_codons": quality["aligned_codons"],
                         "eligible_codons_preflight": quality["eligible_codons_preflight"]}
    finally:
        database.close()


def input_pairs(path: Path, config: RunConfig, workspace: Path) -> Iterator[AlignedPair]:
    for pair, _ in input_pair_records(path, config, workspace):
        yield pair


def collect_input_errors(input_path, config=None, *, workspace=None, max_errors=1000) -> dict:
    """Collect independent input errors without fitting; no recovery row is fit-ready.

    ``workspace`` is an optional parent for temporary disk indexes (created if needed).
    At most ``max_errors`` details are retained, but all recoverable records are
    scanned and counted. A corrupt/unreadable stream cannot be fully scanned.
    """
    config = config or RunConfig()
    if not isinstance(config, RunConfig):
        raise ConfigurationError("config must be a RunConfig")
    if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1:
        raise ConfigurationError("max_errors must be a positive integer")
    if workspace is not None:
        workspace = absolute_path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
    path = absolute_path(input_path)
    result = {"schema": "kaksturbo.input_validation/1", "valid": False,
              "input_path": str(path), "input_format": config.input_format,
              "pairs_seen": 0, "parsed_pairs": 0, "valid_pairs": 0,
              "no_eligible_codon_pairs": 0, "error_count": 0, "errors_truncated": False,
              "errors": [], "alignment_performed": False, "fitting_performed": False}

    def report(error):
        result["error_count"] += 1
        if len(result["errors"]) < max_errors:
            result["errors"].append(error.as_dict())

    with tempfile.TemporaryDirectory(prefix="kaksturbo-validate-", dir=workspace) as temporary:
        for pair, metadata in input_pair_records(path, config, Path(temporary), report_error=report, counters=result):
            result["parsed_pairs"] += 1
            if metadata["eligible_codons_preflight"] == 0:
                result["no_eligible_codon_pairs"] += 1
                report(_located(InputError("Every paired codon contains ambiguity/gaps or a code-1 stop"),
                                metadata["source_location"], pair_id=pair.pair_id, reason="no_eligible_codons"))
            else:
                result["valid_pairs"] += 1
    if not result["pairs_seen"] and not result["error_count"]:
        report(_located(InputError("No aligned gene pairs found"), _location(path, config.input_format)))
    result["errors_truncated"] = result["error_count"] > len(result["errors"])
    result["valid"] = result["error_count"] == 0
    return result


def coerce_pair(value) -> AlignedPair:
    if isinstance(value, AlignedPair):
        pair = value
    elif isinstance(value, dict):
        try:
            pair = AlignedPair(value.get("pair_id", value.get("id")), value["left"], value["right"])
        except KeyError as exc:
            raise InputError("Pair mappings require pair_id, left, right") from exc
    else:
        try:
            pair = AlignedPair(*value)
        except (TypeError, ValueError) as exc:
            raise InputError("Pairs must be AlignedPair, mappings or (pair_id, left, right) triples") from exc
    check_pair(pair)
    return pair


def prepare_chunks(path: Path, config: RunConfig, workspace: Path) -> tuple[list[dict], dict]:
    """Validate the complete input before fitting; duplicate checks never hold all IDs in RAM."""
    from .checkpoint import file_sha256
    workspace.mkdir(parents=True, exist_ok=False)
    chunks, count, masked, masked_examples = [], 0, 0, []
    raw_handle = metadata_handle = None
    try:
        with (workspace / "qc.tsv").open("w", encoding="utf-8", newline="") as qc_handle:
            fields = ["pair_index", "pair_id", "stable_pair_sha256", "aligned_nt", "aligned_codons",
                      "eligible_codons_preflight", "ambiguous_codons", "gap_codons", "stop_codons", "status"]
            writer = csv.DictWriter(qc_handle, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for pair, source in input_pair_records(path, config, workspace):
                quality = source.pop("_qc")
                fingerprint = pair_identity(pair)
                if count % config.chunk_pairs == 0:
                    if raw_handle is not None:
                        raw_handle.close(); metadata_handle.close()
                    index = len(chunks)
                    raw_path = workspace / f"chunk_{index:06d}.axt"
                    metadata_path = workspace / f"chunk_{index:06d}.pairs.jsonl"
                    chunks.append({"index": index, "start": count, "count": 0,
                                   "input": str(raw_path), "pairs": str(metadata_path), "status": "pending", "attempts": []})
                    raw_handle = raw_path.open("wb")
                    metadata_handle = metadata_path.open("w", encoding="utf-8", newline="\n")
                raw_handle.write(pair_record(pair))
                json.dump({"pair_index": count, "pair_id": pair.pair_id, "stable_pair_sha256": fingerprint,
                           **source}, metadata_handle)
                metadata_handle.write("\n")
                writer.writerow({key: ({"pair_index": count, "pair_id": pair.pair_id,
                                       "stable_pair_sha256": fingerprint}.get(key, quality.get(key))) for key in fields})
                chunks[-1]["count"] += 1
                count += 1
                masked += quality["status"] == "no_eligible_codons"
                if quality["status"] == "no_eligible_codons" and len(masked_examples) < 25:
                    masked_examples.append({"pair_id": pair.pair_id, "pair_index": count - 1,
                                            "reason": "Every paired codon contains ambiguity/gaps or a code-1 stop"})
        if raw_handle is not None:
            raw_handle.close(); metadata_handle.close()
            raw_handle = metadata_handle = None
        if not count:
            raise InputError("No aligned gene pairs found")
        for chunk in chunks:
            chunk["input_sha256"] = file_sha256(Path(chunk["input"]))
            chunk["pairs_sha256"] = file_sha256(Path(chunk["pairs"]))
        return chunks, {"pairs": count, "no_eligible_codon_pairs": masked,
                        "no_eligible_pair_examples": masked_examples,
                        "qc_path": str(workspace / "qc.tsv"), "qc_sha256": file_sha256(workspace / "qc.tsv")}
    finally:
        if raw_handle is not None:
            raw_handle.close(); metadata_handle.close()
