"""Public file/iterable API, chunk execution, sealed outputs and restart support."""
from __future__ import annotations
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from ._paths import absolute_path
import shutil
import sys
import tempfile
import time
import uuid
from .checkpoint import atomic_json, contained, file_sha256, load_json, run_lock, utc_now, verify_hashes
from .errors import ConfigurationError, EngineError, InputError, IntegrityError, KaKsTurboError
from .io import coerce_pair, pair_record, prepare_chunks
from .models import MODEL_ORDER, RunConfig, RunResult, VERSION
from .progress import RunProgress, status

CHECKPOINT_SCHEMA = "kaksturbo.checkpoint/2"
MANIFEST_SCHEMA = "kaksturbo.run_manifest/2"


def _engine():
    from . import engine
    return engine


def _identity(config: RunConfig) -> dict:
    try:
        native = _engine().asset_identity(config)
    except KaKsTurboError:
        raise
    except Exception as exc:
        raise IntegrityError(str(exc), details={"engine_error_type": type(exc).__name__}) from exc
    directory = absolute_path(__file__).parent
    public = {path.name: file_sha256(path) for path in sorted(directory.glob("*.py"))}
    return {"schema": "kaksturbo.product_identity/1", "version": VERSION,
            "public_python_sha256": public, "engine": native}


def doctor(config: RunConfig | None = None) -> dict:
    """Inspect installation/runtime availability without performing a fit."""
    return _engine().doctor(config)


def _input_identity(path: Path, config: RunConfig) -> dict:
    if not path.is_file():
        raise InputError(f"Input file does not exist: {path}")
    result = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    for key in ("pairs_manifest", "input_metadata"):
        if not getattr(config, key):
            continue
        manifest = Path(getattr(config, key))
        if not manifest.is_file():
            raise InputError(f"Pairs manifest does not exist: {manifest}")
        result[key] = {"path": str(manifest), "sha256": file_sha256(manifest), "bytes": manifest.stat().st_size}
    return result


def _assert_input(identity: dict, config: RunConfig):
    if _input_identity(Path(identity["path"]), config) != identity:
        raise IntegrityError("Input or pair manifest changed; this run cannot be resumed or sealed")


def _relative(output: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(output.resolve())).replace("\\", "/")
    except ValueError as exc:
        raise EngineError(f"Engine output escaped this run directory: {path}") from exc


def _initial_output(output_dir) -> Path:
    output = absolute_path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ConfigurationError(f"Output already exists; choose a new directory or use resume: {output}") from exc
    (output / ".work").mkdir()
    return output


def fit_file(input_path, output_dir, config: RunConfig | None = None, *, progress=None) -> RunResult:
    """Fit every ordered directed pair in an AXT or explicit aligned-FASTA input.

    Preflight validates the entire input before native fitting starts. Scientific
    raw chunk files are retained, while public JSONL records carry separate global
    metadata. A failed/interrupted run can be continued with :func:`resume`.
    """
    config = RunConfig() if config is None else config
    if not isinstance(config, RunConfig):
        raise ConfigurationError("config must be a RunConfig")
    from .input_metadata import resolve_input
    path, config = resolve_input(input_path, config)
    # Fail before creating a run for an absent input or unavailable installation.
    identity = _input_identity(path, config)
    product = _identity(config)
    output = _initial_output(output_dir)
    with run_lock(output):
        return _start_locked(path, output, config, identity, product, progress)


def fit_pairs(pairs, output_dir, config: RunConfig | None = None, *, progress=None) -> RunResult:
    """Fit a streaming iterable of AlignedPair, mappings or (id,left,right) triples."""
    config = RunConfig() if config is None else config
    if not isinstance(config, RunConfig):
        raise ConfigurationError("config must be a RunConfig")
    if config.input_format != "axt" or config.pairs_manifest is not None or config.input_metadata is not None:
        raise ConfigurationError("fit_pairs accepts already aligned pairs; configure input_format='axt'")
    product = _identity(config)
    output = _initial_output(output_dir)
    with run_lock(output):
        path = output / ".work" / "provided_pairs.axt"
        try:
            with path.open("wb") as handle:
                for value in pairs:
                    handle.write(pair_record(coerce_pair(value)))
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as exc:
            atomic_json(output / "materialization_failure.json", {
                "schema": "kaksturbo.materialization_failure/1", "at": utc_now(),
                "error_type": type(exc).__name__, "message": str(exc),
                "resumable": False, "scope": "Iterable input was not fully materialized; generator cannot be reconstructed"})
            raise
        return _start_locked(path, output, config, _input_identity(path, config), product, progress)


def _start_locked(path: Path, output: Path, config: RunConfig, identity: dict, product: dict, progress=None) -> RunResult:
    state = {"schema": CHECKPOINT_SCHEMA, "version": VERSION, "run_id": uuid.uuid4().hex,
             "created_at": utc_now(), "status": "preflight", "config": config.to_dict(),
             "input": identity, "product_identity": product, "chunks": [], "events": [],
             "elapsed_seconds_completed_sessions": 0.0}
    atomic_json(output / "checkpoint.json", state)
    return _execute(output, state, config, progress)


def resume(output_dir, config: RunConfig | None = None, *, progress=None, max_seconds=None,
           input_path=None, pairs_manifest=None, input_metadata=None) -> RunResult:
    """Continue a checkpoint after verifying full input, assets and sealed chunks."""
    if config is not None and not isinstance(config, RunConfig):
        raise ConfigurationError("config must be a RunConfig")
    output = absolute_path(output_dir)
    if not output.is_dir():
        raise ConfigurationError(f"Run directory does not exist: {output}")
    with run_lock(output):
        state = load_json(output / "checkpoint.json")
        if state.get("schema") != CHECKPOINT_SCHEMA or state.get("version") != VERSION:
            raise IntegrityError("Unsupported checkpoint schema or product version")
        stored = RunConfig(**state["config"])
        if config is not None and config.to_dict() != stored.to_dict():
            raise IntegrityError("Resume configuration differs from the original run")
        changes = {}
        if max_seconds is not None:
            candidate = replace(stored, max_seconds=max_seconds)
            if candidate.max_seconds < stored.max_seconds:
                raise ConfigurationError("Resume can only increase the per-chunk time limit")
            changes["max_seconds"] = candidate.max_seconds
        for key, value in (("pairs_manifest", pairs_manifest), ("input_metadata", input_metadata)):
            if value is not None:
                if not getattr(stored, key):
                    raise IntegrityError("Resume cannot add a new input dependency: " + key)
                changes[key] = str(absolute_path(value))
        effective = replace(stored, **changes)
        path = absolute_path(input_path) if input_path is not None else Path(state["input"]["path"])
        identity = _input_identity(path, effective)
        def content_only(item):
            return {key: content_only(value) if isinstance(value, dict) else value
                    for key, value in item.items() if key != "path"}
        if content_only(identity) != content_only(state["input"]):
            raise IntegrityError("Relocated input/metadata bytes differ from the original run")
        if _identity(stored) != state["product_identity"]:
            raise IntegrityError("Product code or engine assets changed; resume requires the original version and assets")
        _verify_prepared(output, state)
        for chunk in state["chunks"]:
            if chunk["status"] == "complete":
                verify_hashes(output, chunk["artifacts_sha256"])
        if state.get("status") == "complete":
            verify_hashes(output, state["final_artifacts_sha256"])
            manifest = load_json(output / "run_manifest.json")
            return RunResult(output, manifest)
        if effective != stored or identity != state["input"]:
            state["events"].append({"at": utc_now(), "event": "operational_resume_override",
                                    "previous_config": state["config"], "previous_input": state["input"],
                                    "config": effective.to_dict(), "input": identity})
            state["config"] = effective.to_dict()
            state["input"] = identity
            atomic_json(output / "checkpoint.json", state)
        return _execute(output, state, effective, progress)


def _verify_prepared(output: Path, state: dict):
    for chunk in state.get("chunks", []):
        verify_hashes(output, {chunk["input"]: chunk["input_sha256"], chunk["pairs"]: chunk["pairs_sha256"]})
    if state.get("preflight"):
        verify_hashes(output, {state["preflight"]["qc_path"]: state["preflight"]["qc_sha256"]})


def _execute(output: Path, state: dict, config: RunConfig, progress=None) -> RunResult:
    started = time.perf_counter()
    checkpoint = output / "checkpoint.json"
    observer = RunProgress(output, state, progress)
    try:
        if not state["chunks"]:
            _assert_input(state["input"], config)
            attempt = output / ".work" / ("preflight_" + uuid.uuid4().hex)
            chunks, preflight = prepare_chunks(Path(state["input"]["path"]), config, attempt)
            _assert_input(state["input"], config)
            for chunk in chunks:
                chunk["input"] = _relative(output, Path(chunk["input"]))
                chunk["pairs"] = _relative(output, Path(chunk["pairs"]))
            preflight["qc_path"] = _relative(output, Path(preflight["qc_path"]))
            state.update(chunks=chunks, preflight=preflight, status="running")
            atomic_json(checkpoint, state)
            observer.update(state)
        if state["preflight"]["no_eligible_codon_pairs"]:
            report = {"schema": "kaksturbo.input_validation/1", "valid": False,
                      "scientific_fitting_performed": False, "input": state["input"], **state["preflight"]}
            atomic_json(output / "validation.json", report)
            raise InputError("One or more pairs have no eligible codons after the historical paired filter; inspect QC and quarantine or correct those input records", details={
                "count": state["preflight"]["no_eligible_codon_pairs"],
                "pair_examples": state["preflight"]["no_eligible_pair_examples"],
                "qc": state["preflight"]["qc_path"], "validation_report": "validation.json"})
        _verify_prepared(output, state)
        from . import _parallel_aggregate
        _parallel_aggregate.prepare(state)
        atomic_json(checkpoint, state)
        for chunk in state["chunks"]:
            if chunk["status"] == "complete":
                continue
            chunk_dir = output / ".work" / f"chunk_{chunk['index']:06d}" / ("attempt_" + uuid.uuid4().hex)
            chunk_dir.parent.mkdir(parents=True, exist_ok=True)
            attempt = {"directory": _relative(output, chunk_dir), "started_at": utc_now(), "status": "running"}
            # A previous process may have died before recording its failure.
            for old in chunk["attempts"]:
                if old["status"] == "running":
                    old.update(status="interrupted", recovered_at=utc_now())
            chunk["attempts"].append(attempt)
            chunk["status"] = "running"
            state["status"] = "running"
            atomic_json(checkpoint, state)
            observer.update(state)
            try:
                returned = _engine().run_chunk(contained(output, chunk["input"]), chunk_dir, config)
            except KaKsTurboError:
                raise
            except Exception as exc:
                raise EngineError(str(exc), details={"chunk_index": chunk["index"], "engine_error_type": type(exc).__name__}) from exc
            artifacts = _normalize_engine(output, chunk, returned)
            _check_chunk(output, chunk, artifacts)
            hashes = {_relative(output, path): file_sha256(path) for path in chunk_dir.rglob("*") if path.is_file()}
            if not hashes:
                raise EngineError("Engine returned an empty artifact directory")
            attempt.update(status="complete", completed_at=utc_now())
            chunk.update(status="complete", outputs=artifacts, artifacts_sha256=hashes,
                         backend=returned.get("backend"), quality=returned.get("quality"))
            atomic_json(checkpoint, state)
            observer.update(state)
        _assert_input(state["input"], config)
        if _identity(config) != state["product_identity"]:
            raise IntegrityError("Product code or engine assets changed during the run; final results cannot be sealed")
        state["status"] = "aggregating"
        atomic_json(checkpoint, state)
        observer.update(state)
        final = _aggregate(output, state)
        from .quality import write_quality_summary
        quality = write_quality_summary(output, expected_pairs=state["preflight"]["pairs"])
        final = list(final) + ["quality_summary.json", "quality_summary.txt"]
        elapsed = state["elapsed_seconds_completed_sessions"] + time.perf_counter() - started
        manifest = {"schema": MANIFEST_SCHEMA, "version": VERSION, "complete": True,
                    "run_id": state["run_id"], "created_at": state["created_at"], "completed_at": utc_now(),
                    "mode": config.mode, "method": "MA", "models": list(MODEL_ORDER), "model_count": 14,
                    "genetic_code": 1, "count": state["preflight"]["pairs"], "config": config.to_dict(),
                    "input": state["input"], "product_identity": state["product_identity"],
                    "chunks": state["chunks"], "preflight": state["preflight"],
                    "backends_used": sorted({c["backend"] for c in state["chunks"] if c.get("backend")}),
                    "elapsed_seconds_completed_sessions": elapsed,
                    "timing_scope": "API sessions include input hashing, preflight, chunk input/output, transfers, fitting and aggregation; interrupted sessions cannot claim a complete timing",
                    "scientific_identity_scope": "Native chunk fields are preserved in record envelopes; global_pair_index and stable_pair_sha256 identify ordered directed input; native fingerprints remain chunk-local",
                    "statistical_scope": "Ka/Ks is not a significance test; no positive-selection P value or global-optimum certificate is inferred",
                    "result_schema": "kaksturbo.core_result/1",
                    "native_result": "native_result.tsv",
                    "quality_summary": quality,
                    "artifacts_sha256": {name: file_sha256(output / name) for name in final}}
        atomic_json(output / "run_manifest.json", manifest)
        state.update(status="complete", completed_at=manifest["completed_at"],
                     elapsed_seconds_completed_sessions=elapsed,
                     final_artifacts_sha256={**manifest["artifacts_sha256"], "run_manifest.json": file_sha256(output / "run_manifest.json")})
        atomic_json(checkpoint, state)
        observer.update(state)
        return RunResult(output, manifest)
    except BaseException as exc:
        state["elapsed_seconds_completed_sessions"] += time.perf_counter() - started
        state["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        state["events"].append({"at": utc_now(), "error_type": type(exc).__name__, "message": str(exc),
                                "status": state["status"]})
        for chunk in state.get("chunks", []):
            if chunk["status"] == "running":
                chunk["status"] = "pending"
                chunk["attempts"][-1].update(status=state["status"], completed_at=utc_now(), error=str(exc))
        atomic_json(checkpoint, state)
        observer.update(state)
        raise
    finally:
        observer.close()


def _normalize_engine(output: Path, chunk: dict, returned: dict) -> dict:
    if not isinstance(returned, dict) or returned.get("count") != chunk["count"]:
        raise EngineError("Engine count differs from the requested chunk")
    paths = returned.get("paths", returned)
    result = {}
    for key, filename in (("models", "models.json"), ("pair_statistics", "pair_statistics.json"),
                          ("result", "result.tsv"), ("run_manifest", "run_manifest.json")):
        value = paths.get(key, paths.get(filename))
        if value is None:
            raise EngineError(f"Engine omitted its required artifact: {filename}")
        path = Path(value).resolve()
        if not path.is_file():
            raise EngineError(f"Engine artifact does not exist: {path}")
        attempt_root = contained(output, chunk["attempts"][-1]["directory"])
        try:
            path.relative_to(attempt_root)
        except ValueError as exc:
            raise EngineError(f"Engine returned a file outside its owned attempt: {path}") from exc
        result[key] = _relative(output, path)
    return result


def _metadata(output: Path, chunk: dict) -> list[dict]:
    with contained(output, chunk["pairs"]).open(encoding="utf-8") as handle:
        result = [json.loads(line) for line in handle]
    if len(result) != chunk["count"]:
        raise IntegrityError("Prepared pair metadata count differs")
    return result


def _pair_name(record: dict):
    return record.get("pair", record.get("pair_id"))


def _read_native(output: Path, paths: dict):
    with contained(output, paths["models"]).open(encoding="utf-8") as handle:
        models = json.load(handle)
    with contained(output, paths["pair_statistics"]).open(encoding="utf-8") as handle:
        statistics = json.load(handle)
    with contained(output, paths["result"]).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    return models, statistics, fields, rows


def _check_chunk(output: Path, chunk: dict, paths: dict):
    metadata = _metadata(output, chunk)
    models, statistics, fields, rows = _read_native(output, paths)
    if not isinstance(models, list) or len(models) != 14 * chunk["count"]:
        raise EngineError("Engine must export exactly fourteen candidate records per pair")
    if not isinstance(statistics, list) or len(statistics) != chunk["count"] or len(rows) != chunk["count"]:
        raise EngineError("Engine pair-statistics/result count differs from requested pairs")
    if not fields or len(fields) != len(set(fields)) or "Sequence" not in fields or "Method" not in fields:
        raise EngineError("Engine result TSV header is invalid")
    for local, meta in enumerate(metadata):
        name = meta["pair_id"]
        if rows[local].get("Sequence") != name or rows[local].get("Method") != "MA":
            raise EngineError(f"Engine changed requested order, ID or method: {name}")
        if None in rows[local] or any(value is None for value in rows[local].values()):
            raise EngineError(f"Engine result TSV is malformed: {name}")
        if _pair_name(statistics[local]) != name:
            raise EngineError(f"Engine pair-statistics ID/order differs: {name}")
        for model_index, model_name in enumerate(MODEL_ORDER):
            record = models[local * 14 + model_index]
            if _pair_name(record) != name or record.get("model") != model_name:
                raise EngineError(f"Engine changed one of the fourteen model slots: {name}, {model_name}")
    manifest = load_json(contained(output, paths["run_manifest"]))
    if manifest.get("complete") is not True:
        raise EngineError("Engine run manifest does not certify completion")
    return metadata, models, statistics, fields, rows


def _aggregate(output: Path, state: dict) -> list[str]:
    from . import _parallel_aggregate
    return _parallel_aggregate.aggregate(sys.modules[__name__], output, state)


def _aggregate_serial(output: Path, state: dict, *, destination=None) -> list[str]:
    from .results import CORE_FIELDS, core_row, format_row, write_data_dictionary
    destination = output if destination is None else Path(destination)
    staging = destination / ".work" / (("aggregation_" + uuid.uuid4().hex) if destination == output else ("s" + uuid.uuid4().hex[:12]))
    staging.mkdir()
    names = ["result.tsv", "results.jsonl", "native_result.tsv", "native_results.jsonl",
             "pair_statistics.jsonl", "models.jsonl", "models.tsv", "qc.tsv", "data_dictionary.json"]
    handles = {name: (staging / name).open("w", encoding="utf-8", newline="") for name in names
               if name not in {"qc.tsv", "data_dictionary.json"}}
    header = None
    result_writer = csv.DictWriter(handles["result.tsv"], CORE_FIELDS, delimiter="\t", lineterminator="\n")
    result_writer.writeheader()
    model_writer = csv.DictWriter(handles["models.tsv"], ["pair_index", "pair_id", "stable_pair_sha256", "global_slot",
        "model", "chunk_index", "native_slot", "status", "quality_passed", "record_json"], delimiter="\t", lineterminator="\n")
    model_writer.writeheader()
    try:
        for chunk in state["chunks"]:
            verify_hashes(output, chunk["artifacts_sha256"])
            metadata, models, statistics, fields, rows = _check_chunk(output, chunk, chunk["outputs"])
            if header is None:
                header = fields
                native_writer = csv.DictWriter(handles["native_result.tsv"], fields, delimiter="\t", lineterminator="\n")
                native_writer.writeheader()
            elif fields != header:
                raise EngineError("Result schemas differ across chunks")
            for local, meta in enumerate(metadata):
                envelope = {"schema": "kaksturbo.result_record/1", "global_pair_index": meta["pair_index"],
                            "pair_id": meta["pair_id"], "stable_pair_sha256": meta["stable_pair_sha256"],
                            "chunk_index": chunk["index"]}
                native_writer.writerow(rows[local])
                handles["native_results.jsonl"].write(json.dumps({**envelope, "record": rows[local]}, allow_nan=False) + "\n")
                core = core_row(rows[local], statistics[local], {**meta, "backend": chunk.get("backend")},
                                state["config"], model_records=models[local * 14:(local + 1) * 14])
                result_writer.writerow(format_row(core))
                handles["results.jsonl"].write(json.dumps({**envelope, "schema": "kaksturbo.result_record/2",
                                                          "record": core}, allow_nan=False) + "\n")
                handles["pair_statistics.jsonl"].write(json.dumps({**envelope, "schema": "kaksturbo.pair_statistics_record/1",
                                                                  "record": statistics[local]}, allow_nan=False) + "\n")
                for model_index in range(14):
                    record = models[local * 14 + model_index]
                    global_slot = meta["pair_index"] * 14 + model_index
                    handles["models.jsonl"].write(json.dumps({**envelope, "schema": "kaksturbo.model_record/1",
                        "global_slot": global_slot, "record": record}, allow_nan=False) + "\n")
                    model_writer.writerow({"pair_index": meta["pair_index"], "pair_id": meta["pair_id"],
                        "stable_pair_sha256": meta["stable_pair_sha256"], "global_slot": global_slot,
                        "model": MODEL_ORDER[model_index], "chunk_index": chunk["index"], "native_slot": record.get("slot", ""),
                        "status": record.get("status") or "NA",
                        "quality_passed": "NA" if record.get("quality_passed") is None else record["quality_passed"],
                        "record_json": json.dumps(record, allow_nan=False, separators=(",", ":"))})
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    shutil.copyfile(contained(output, state["preflight"]["qc_path"]), staging / "qc.tsv")
    write_data_dictionary(staging / "data_dictionary.json")
    for name in names:
        os.replace(staging / name, destination / name)
    return names


def validate(input_path, config: RunConfig | None = None, output_dir=None, *, max_errors=1000) -> dict:
    """Collect input problems with source locations; never initialize a fitting engine."""
    from .io import collect_input_errors
    config = RunConfig() if config is None else config
    if not isinstance(config, RunConfig):
        raise ConfigurationError("config must be a RunConfig")
    from .input_metadata import resolve_input
    path, config = resolve_input(input_path, config)
    identity = _input_identity(path, config)
    def inspect(directory):
        report = collect_input_errors(path, config, workspace=directory / "scan", max_errors=max_errors)
        if report["valid"]:
            _, preflight = prepare_chunks(path, config, directory / "preflight")
            shutil.copyfile(Path(preflight.pop("qc_path")), directory / "qc.tsv")
            preflight.pop("qc_sha256", None)
            report.update(preflight)
        _assert_input(identity, config)
        return {**report, "schema": "kaksturbo.input_validation/2", "version": VERSION,
                "input": identity, "config": config.to_dict(), "scientific_fitting_performed": False,
                "AXT_wrapping_policy": "Body concatenation and midpoint split; IDs, sequence bytes and case preserved",
                "FASTA_ID_policy": "First whitespace-delimited header token; directed pairs only from manifest"}

    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix=".kaksturbo-validation-", dir=Path.cwd()) as temporary:
            return inspect(Path(temporary))
    output = _initial_output(output_dir)
    with run_lock(output):
        result = inspect(output / ".work")
        if (output / ".work/qc.tsv").is_file():
            shutil.copyfile(output / ".work/qc.tsv", output / "qc.tsv")
            result["qc_path"] = str(output / "qc.tsv")
        fields = ["file", "line", "line_end", "record", "pair_id", "error", "message", "details_json"]
        with (output / "errors.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for error in result.get("errors", []):
                details = error.get("details", {})
                location = details.get("source_location", {})
                writer.writerow({"file": location.get("path"), "line": location.get("line"),
                                 "line_end": location.get("line_end"), "record": location.get("record"),
                                 "pair_id": details.get("pair_id"), "error": error.get("error"), "message": error.get("message"),
                                 "details_json": json.dumps(details, ensure_ascii=False)})
        result["errors_path"] = str(output / "errors.tsv")
        atomic_json(output / "validation.json", result)
        return result
