"""Versioned, typed result rows; this module never estimates scientific values."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from .errors import EngineError, IntegrityError
from .models import MODEL_ORDER

RESULT_SCHEMA = "kaksturbo.core_result/1"
CORE_FIELDS = ("pair_id", "seq1_id", "seq2_id", "method", "mode", "backend",
               "Ka", "Ks", "Ka_Ks", "aligned_codons", "eligible_codons", "observed_SNP",
               "status", "reason", "lnL", "best_model", "iteration_capped_models", "boundary_models")
FLOAT_FIELDS = frozenset(("Ka", "Ks", "Ka_Ks", "lnL"))
INT_FIELDS = frozenset(("aligned_codons", "eligible_codons", "observed_SNP"))
LIST_FIELDS = frozenset(("iteration_capped_models", "boundary_models"))
REQUIRED_TEXT = frozenset(("pair_id", "method", "mode", "status"))
TEXT_FIELDS = frozenset(CORE_FIELDS) - FLOAT_FIELDS - INT_FIELDS - LIST_FIELDS
DESCRIPTIONS = {
    "pair_id": "Original ordered request identifier; never split to guess sequence identifiers.",
    "seq1_id": "Explicit left sequence identifier, if supplied by the input format.",
    "seq2_id": "Explicit right sequence identifier, if supplied by the input format.",
    "method": "Requested estimator (MA in this release).",
    "mode": "robust or reproduce; these are distinct estimators.",
    "backend": "Actual execution backend, cpu or gpu; null when not recorded.",
    "Ka": "Reported nonsynonymous substitutions per nonsynonymous site; copied without recalculation.",
    "Ks": "Reported synonymous substitutions per synonymous site; copied without recalculation.",
    "Ka_Ks": "Reported MA Ka/Ks; never recomputed as the ratio of the displayed Ka and Ks.",
    "aligned_codons": "Input aligned codon count before paired codon filtering, when recorded.",
    "eligible_codons": "Paired codons retained by the code-1 input filter, when recorded.",
    "observed_SNP": "Reported nucleotide differences after codon filtering; not an expected event count.",
    "status": "Native MA availability (available, unidentifiable or unavailable). This is not proof of convergence.",
    "reason": "Recorded explanation and explicit missing-field information; empty if none is recorded.",
    "lnL": "Native reported MA log likelihood. Official/reproduce ML-Score is copied; different estimator objectives are not interchangeable.",
    "best_model": "Native reported candidate selected by the model criterion; null when absent, including historical MA output.",
    "iteration_capped_models": "Candidate names with recorded iteration limits. [] means known none; null means diagnostic unavailable.",
    "boundary_models": "Candidate names with recorded active parameter-bound hits. Requires complete bound diagnostics; [] means known none, null means unavailable. This does not infer significance or MA-bound status.",
}


def _numeric(value, field):
    if value is None or (isinstance(value, str) and value.strip() in ("", "NA", "N/A")):
        return None
    if isinstance(value, bool):
        raise EngineError(f"Boolean is not a result number: {field}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EngineError(f"Invalid numeric result: {field}={value!r}") from exc
    if not math.isfinite(number):
        raise EngineError(f"Nonfinite numeric result: {field}")
    if field in INT_FIELDS:
        if number < 0 or number != int(number):
            raise EngineError(f"Expected a nonnegative integer: {field}")
        return int(number)
    return number


def _native_text(value):
    return None if value in (None, "", "NA", "N/A") else str(value)


def _model_list(value):
    if value is None or value == "NA":
        return None
    if value in ("", "none"):
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)) or any(v not in MODEL_ORDER for v in value) or len(set(value)) != len(value):
        raise EngineError("Malformed candidate-model list")
    return list(value)


def validate_row(row):
    """Return a typed copy with a fixed field set; reject nonfinite or malformed data."""
    if set(row) != set(CORE_FIELDS):
        raise EngineError("Core result fields differ from the versioned schema")
    result = {}
    for field in CORE_FIELDS:
        value = row[field]
        if field in FLOAT_FIELDS | INT_FIELDS:
            result[field] = _numeric(value, field)
        elif field in LIST_FIELDS:
            if value is not None and not isinstance(value, list):
                raise EngineError(f"Expected a typed model list or null: {field}")
            result[field] = _model_list(value)
        else:
            if value is not None and not isinstance(value, str):
                raise EngineError(f"Expected text or null: {field}")
            if field in REQUIRED_TEXT and not value:
                raise EngineError(f"Missing required result field: {field}")
            result[field] = None if field not in REQUIRED_TEXT and value == "" else value
    return result


def core_row(native_row, statistics, metadata, config=None, *, mode=None, model_records=None):
    """Map one native row and existing diagnostics into the stable core schema.

    metadata may include seq1_id/seq2_id, aligned_codons, eligible_codons (or
    eligible_codons_preflight), and actual backend. No identifiers are inferred.
    model_records optionally contains the pair's complete 14 native records.
    """
    statistics, metadata = statistics or {}, metadata or {}
    if "record" in statistics:
        statistics = statistics["record"]
    cfg = config if isinstance(config, dict) else vars(config) if config is not None and not isinstance(config, str) else {}
    chosen_mode = mode or (config if isinstance(config, str) else cfg.get("mode")) or statistics.get("mode")
    if chosen_mode not in ("robust", "reproduce"):
        raise EngineError("Core results require the recorded robust/reproduce mode")
    pair = native_row.get("Sequence")
    if not isinstance(pair, str) or not pair:
        raise EngineError("Native result lacks its pair identifier")
    for recorded in (metadata.get("pair_id"), statistics.get("pair", statistics.get("pair_id"))):
        if recorded is not None and recorded != pair:
            raise EngineError("Native result, metadata and statistics pair identifiers differ")
    ma = statistics.get("MA") or {}
    numbers = {field: _numeric(native_row.get(native), field) for field, native in (
        ("Ka", "Ka"), ("Ks", "Ks"), ("Ka_Ks", "Ka/Ks"),
        ("lnL", "lnL" if "lnL" in native_row else "ML-Score"))}
    status = _native_text(native_row.get("Status")) or ma.get("status")
    if status is None:
        status = "available" if numbers["Ka_Ks"] is not None else "unavailable"
    reasons = []
    if ma.get("reason"):
        reasons.append(str(ma["reason"]))
    snp_raw = native_row.get("Observed-SNP", native_row.get("Substitutions", statistics.get("masked_observed_SNP")))
    snp = _numeric(snp_raw, "observed_SNP")
    if chosen_mode == "robust" and status == "unidentifiable" and snp == 0:
        reasons.insert(0, "zero_observed_substitutions")
    missing = [field for field in ("Ka", "Ks", "Ka_Ks") if numbers[field] is None]
    if missing:
        reasons.append(("historical_reported_NA:" if chosen_mode == "reproduce" else "reported_NA:") + ",".join(missing))
    if status != "available" and not reasons:
        reasons.append("native_status:" + status)

    qc = metadata.get("qc", {})
    aligned = metadata.get("aligned_codons", qc.get("aligned_codons"))
    eligible = metadata.get("eligible_codons", metadata.get("eligible_codons_preflight", qc.get("eligible_codons_preflight")))
    statistic_length = statistics.get("codon_count")
    if eligible is None:
        eligible = statistic_length
    elif statistic_length is not None and _numeric(eligible, "eligible_codons") != _numeric(statistic_length, "eligible_codons"):
        raise EngineError("Preflight and fitted eligible codon counts differ")
    # Old historical outputs can supply the cleaned nucleotide length only.
    if eligible is None and _native_text(native_row.get("Length")) is not None:
        nt = _numeric(native_row["Length"], "eligible_codons")
        if nt % 3:
            raise EngineError("Native cleaned nucleotide length is not codon-aligned")
        eligible = nt // 3
    capped = boundary = None
    if chosen_mode == "robust":
        if "Iteration-capped-models" in native_row:
            capped = _model_list(native_row["Iteration-capped-models"])
        elif "iteration_capped_models" in statistics:
            capped = _model_list(statistics["iteration_capped_models"])
        if model_records is not None:
            records = [item.get("record", item) for item in model_records]
            if len(records) != 14 or {r.get("model") for r in records} != set(MODEL_ORDER):
                raise EngineError("Boundary summaries require all fourteen distinct model records")
            if any(r.get("pair", r.get("pair_id")) != pair for r in records):
                raise EngineError("Model diagnostics belong to another pair")
            # Native failed/nonfinite endpoints can carry an empty placeholder
            # list. It is not evidence that a usable endpoint had no bound hits.
            endpoint_known = all("result18" not in r or (isinstance(r["result18"], list)
                                 and len(r["result18"]) == 18
                                 and all(type(v) in (int, float) and math.isfinite(v) for v in r["result18"]))
                                 for r in records)
            if endpoint_known and all(isinstance(r.get("active_parameter_bound_hits"), list) for r in records):
                names = {r["model"] for r in records if r["active_parameter_bound_hits"]}
                boundary = [name for name in MODEL_ORDER if name in names]
            if capped is None and all(type(r.get("iteration_limit_reached")) is bool for r in records):
                names = {r["model"] for r in records if r["iteration_limit_reached"]}
                capped = [name for name in MODEL_ORDER if name in names]
    backend = metadata.get("backend", cfg.get("backend"))
    if backend == "auto":
        backend = None
    row = dict(pair_id=pair, seq1_id=metadata.get("seq1_id"), seq2_id=metadata.get("seq2_id"),
               method=native_row.get("Method"), mode=chosen_mode, backend=backend,
               **numbers, aligned_codons=_numeric(aligned, "aligned_codons"),
               eligible_codons=_numeric(eligible, "eligible_codons"), observed_SNP=snp,
               status=status, reason="; ".join(reasons) or None,
               best_model=_native_text(native_row.get("Best-model", native_row.get("Model"))),
               iteration_capped_models=capped, boundary_models=boundary)
    return validate_row(row)


def format_row(row):
    """TSV fields: round-trip floats, nullable text empty, numeric/list null NA."""
    row = validate_row(row)
    result = {}
    for field, value in row.items():
        if value is None:
            result[field] = "" if field in TEXT_FIELDS else "NA"
        elif field in FLOAT_FIELDS:
            result[field] = format(value, ".17g")
        elif field in LIST_FIELDS:
            result[field] = json.dumps(value, separators=(",", ":"))
        else:
            result[field] = str(value)
    return result


def parse_row(row):
    """Parse the exact versioned TSV schema into JSON-compatible native types."""
    if set(row) != set(CORE_FIELDS) or any(value is None for value in row.values()):
        raise IntegrityError("Malformed core result TSV row")
    parsed = {}
    try:
        for field, value in row.items():
            if field in LIST_FIELDS:
                parsed[field] = None if value == "NA" else json.loads(value)
            elif field in TEXT_FIELDS:
                parsed[field] = value if value or field in REQUIRED_TEXT else None
            else:
                parsed[field] = _numeric(value, field)
        return validate_row(parsed)
    except (EngineError, ValueError, TypeError) as exc:
        raise IntegrityError(f"Invalid core result row: {exc}") from exc


def iter_core_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(CORE_FIELDS):
            raise IntegrityError("Unsupported core result TSV header")
        for row in reader:
            yield parse_row(row)


def data_dictionary():
    return {"schema": RESULT_SCHEMA, "fields": [
        {"name": field, "type": "number" if field in FLOAT_FIELDS else "integer" if field in INT_FIELDS
         else "array[string]" if field in LIST_FIELDS else "string", "nullable": field not in REQUIRED_TEXT,
         "description": DESCRIPTIONS[field]} for field in CORE_FIELDS],
        "missing_values": {"JSON": "null", "TSV_numeric_or_list": "NA", "TSV_nullable_text": "empty field",
                           "empty_model_list": "[] (known none); NA/null means unknown"},
        "precision": "FP64 values use .17g TSV serialization; historical printed values retain their original information content.",
        "interpretation": "Availability is not optimizer convergence. Ka/Ks > 1 is not a positive-selection significance test. Native results and model diagnostics retain estimator-specific information."}


def write_data_dictionary(path):
    from .checkpoint import atomic_json
    path = Path(path)
    atomic_json(path, data_dictionary())
    return path
