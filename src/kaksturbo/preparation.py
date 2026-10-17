"""Prepare already codon-aligned inputs; no alignment or fitting is performed."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from ._paths import absolute_path
import tempfile
from .checkpoint import atomic_json, file_sha256
from .errors import ConfigurationError, InputError
from .io import collect_input_errors, input_pair_records, pair_identity, pair_record
from .models import RunConfig, VERSION


def _new_directory(output_dir):
    output = absolute_path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ConfigurationError(f"Output already exists; choose a new directory: {output}") from exc
    return output


def prepare_aligned(input_path, output_dir, config: RunConfig | None = None) -> dict:
    """Normalize aligned AXT or aligned FASTA + explicit pairs to ``pairs.axt``.

    The caller supplies an existing codon alignment. Structural validation can
    detect unequal lengths, invalid symbols and incomplete codons, but cannot
    establish biological homology or infer the reading frame. No alignment,
    reverse complementation, uppercasing, trimming or codon removal occurs.
    A new output directory is required; invalid input only leaves its validation
    report. The success marker is ``preparation.json`` with ``complete=true``.
    """
    config = config or RunConfig()
    if not isinstance(config, RunConfig):
        raise ConfigurationError("config must be a RunConfig")
    from .input_metadata import resolve_input
    path, config = resolve_input(input_path, config)
    output = _new_directory(output_dir)
    sources = [path] + [Path(value) for value in (config.pairs_manifest, config.input_metadata) if value]
    source_hashes = {str(source): file_sha256(source) for source in sources if source.is_file()}
    validation = collect_input_errors(path, config, workspace=output)
    atomic_json(output / "validation.json", validation)
    if not validation["valid"]:
        raise InputError("Input preparation failed; fix the errors in validation.json. Already codon-aligned input is required.",
                         details={"validation_path": str(output / "validation.json"),
                                  "error_count": validation["error_count"], "errors": validation["errors"]})
    fields = ["pair_index", "pair_id", "stable_pair_sha256", "aligned_nt", "aligned_codons",
              "eligible_codons_preflight", "ambiguous_codons", "gap_codons", "stop_codons", "status"]
    names = ("pairs.axt", "pairs.metadata.jsonl", "qc.tsv")
    count = 0
    # Stage every file under the new directory so neither a malformed later
    # record nor interruption exposes an apparently complete normalized AXT.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
        stage = Path(temporary)
        with (stage / names[0]).open("wb") as axt, (stage / names[1]).open("w", encoding="utf-8", newline="\n") as metadata, \
                (stage / names[2]).open("w", encoding="utf-8", newline="") as quality_file:
            writer = csv.DictWriter(quality_file, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for pair, source in input_pair_records(path, config, stage):
                quality = source.pop("_qc")
                if quality["status"] != "valid":
                    raise InputError(f"No eligible codons after the historical paired filter: {pair.pair_id}",
                                     details={"pair_id": pair.pair_id, "source_location": source["source_location"]})
                identity = {"pair_index": count, "pair_id": pair.pair_id, "stable_pair_sha256": pair_identity(pair)}
                axt.write(pair_record(pair))
                metadata.write(json.dumps({**identity, **source}, ensure_ascii=True) + "\n")
                writer.writerow({key: {**quality, **identity}.get(key) for key in fields})
                count += 1
        if not count:
            raise InputError("No aligned gene pairs found during preparation")
        for source in sources:
            if not source.is_file() or file_sha256(source) != source_hashes.get(str(source)):
                raise InputError("Input or pair manifest changed during preparation; restart with a new output directory",
                                 details={"path": str(source)})
        for name in names:
            (stage / name).replace(output / name)
    result = {"schema": "kaksturbo.input_preparation/1", "version": VERSION, "complete": True,
              "directory": str(output), "input_path": str(path), "input_format": config.input_format,
              "source_sha256": source_hashes,
              "pairs_manifest": config.pairs_manifest, "pairs": count,
              "alignment_performed": False, "fitting_performed": False,
              "normalization": "AXT line layout only; IDs, directed order, sequence case and symbols retained",
              "axt_path": str(output / names[0]), "metadata_path": str(output / names[1]),
              "qc_path": str(output / names[2]), "validation_path": str(output / "validation.json"),
              "sha256": {name: file_sha256(output / name) for name in names}}
    atomic_json(output / "preparation.json", result)
    return result


_EXAMPLES = {
    "aligned.fasta": ">demo_A already codon-aligned example\natgAAACCCGGG\n>demo_B already codon-aligned example\natgAAGCCTGGG\n",
    "pairs.tsv": "pair_id\tseq1\tseq2\ndemo_A_to_B\tdemo_A\tdemo_B\ndemo_B_to_A\tdemo_B\tdemo_A\n",
    "pairs.template.tsv": "pair_id\tseq1\tseq2\n",
    "pairs.axt": "demo_A_to_B\natgAAACCCGGG\natgAAGCCTGGG\n\ndemo_B_to_A\natgAAGCCTGGG\natgAAACCCGGG\n\n",
    "README.txt": "These tiny inputs illustrate formats, not biological inference or accuracy.\n"
                  "FASTA sequences are already codon-aligned; pairs.tsv explicitly requests both directions.\n"
                  "Fill pairs.template.tsv yourself; pairings are never inferred from names or FASTA order.\n"
                  "Raw, unaligned CDS must be aligned outside this preparation interface first.\n",
}


def write_input_examples(output_dir) -> dict:
    """Write explicit illustrative inputs and an empty pair-table template."""
    output = _new_directory(output_dir)
    for name, content in _EXAMPLES.items():
        with (output / name).open("x", encoding="ascii", newline="\n") as handle:
            handle.write(content)
    return {"schema": "kaksturbo.input_examples/1", "directory": str(output),
            "files": {name: str(output / name) for name in _EXAMPLES},
            "pairing_inferred": False, "alignment_performed": False, "fitting_performed": False}
