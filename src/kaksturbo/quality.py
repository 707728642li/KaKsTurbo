"""Streaming descriptive summaries of saved outputs; never refit or certify optima."""
import json
import math
from pathlib import Path
from .checkpoint import atomic_json
from .errors import IntegrityError
from .models import MODEL_ORDER, VERSION
from .results import iter_core_rows

LIMITATIONS = (
    "Finite Ka/Ks and a passed coordinate gate do not establish optimizer convergence.",
    "The projected normalized gradient threshold is 1e-8 at the saved endpoint; missing diagnostics remain unknown.",
    "A boundary hit or iteration cap is a warning, not proof that the final model-averaged Ka/Ks is wrong.",
    "This summary does not refit models, establish a global optimum, or test positive selection.",
)


def summarize_quality(directory, *, expected_pairs=None):
    directory = Path(directory)
    pairs = dict(total=0, finite_Ka_Ks=0, NA_Ka_Ks=0, iteration_capped=0,
                 cap_diagnostic_unknown=0, boundary=0, boundary_diagnostic_unknown=0)
    models = dict(total=0, finite_endpoint=0, unavailable_endpoint=0, endpoint_unknown=0,
                  coordinate_gate_passed=0, coordinate_gate_failed=0, coordinate_gate_unknown=0,
                  iteration_capped=0, cap_unknown=0, boundary=0, boundary_unknown=0,
                  gradient_within_1e_minus_8=0, gradient_above_1e_minus_8=0, gradient_unknown=0)
    with (directory / "models.jsonl").open(encoding="utf-8") as stream:
        for index, row in enumerate(iter_core_rows(directory / "result.tsv")):
            pairs["total"] += 1
            pairs["finite_Ka_Ks" if row["Ka_Ks"] is not None else "NA_Ka_Ks"] += 1
            for field, yes, unknown in (("iteration_capped_models", "iteration_capped", "cap_diagnostic_unknown"),
                                        ("boundary_models", "boundary", "boundary_diagnostic_unknown")):
                if row[field] is None:
                    pairs[unknown] += 1
                elif row[field]:
                    pairs[yes] += 1
            for name in MODEL_ORDER:
                try:
                    envelope = json.loads(next(stream))
                except (StopIteration, ValueError) as exc:
                    raise IntegrityError(f"Missing/malformed model record at pair {index + 1}: {name}") from exc
                record = envelope.get("record", {})
                if (envelope.get("global_pair_index") != index or envelope.get("pair_id") != row["pair_id"]
                        or record.get("model") != name):
                    raise IntegrityError("Model/result identity or model order mismatch in quality summary")
                models["total"] += 1
                endpoint = record.get("result18")
                if not isinstance(endpoint, list) or len(endpoint) != 18:
                    models["endpoint_unknown"] += 1
                    endpoint_finite = False
                else:
                    endpoint_finite = all(type(x) in (int, float) and math.isfinite(x) for x in endpoint)
                    models["finite_endpoint" if endpoint_finite else "unavailable_endpoint"] += 1
                gate = record.get("quality_passed")
                models["coordinate_gate_passed" if gate is True else "coordinate_gate_failed" if gate is False else "coordinate_gate_unknown"] += 1
                cap = record.get("iteration_limit_reached")
                if type(cap) is not bool:
                    models["cap_unknown"] += 1
                elif cap:
                    models["iteration_capped"] += 1
                hits = record.get("active_parameter_bound_hits")
                if not endpoint_finite or not isinstance(hits, list):
                    models["boundary_unknown"] += 1
                elif hits:
                    models["boundary"] += 1
                diagnostic = record.get("returned_gradient_diagnostic") or {}
                norm = diagnostic.get("max_projected_norm") if isinstance(diagnostic, dict) else None
                if not endpoint_finite or type(norm) not in (int, float) or not math.isfinite(norm) or norm < 0:
                    models["gradient_unknown"] += 1
                else:
                    models["gradient_within_1e_minus_8" if norm <= 1e-8 else "gradient_above_1e_minus_8"] += 1
        if stream.read().strip():
            raise IntegrityError("Extra model records in quality summary")
    if expected_pairs is not None and pairs["total"] != expected_pairs:
        raise IntegrityError("Quality summary pair count differs from the run")
    return {"schema": "kaksturbo.quality_summary/1", "version": VERSION, "pairs": pairs, "models": models,
            "model_count_per_pair": 14, "gradient_threshold": 1e-8,
            "scope": "Descriptive diagnostics from saved result.tsv and models.jsonl; no fitting performed",
            "limitations": list(LIMITATIONS)}


def quality_text(summary):
    lines = ["KaKsTurbo quality summary", ""]
    for category in ("pairs", "models"):
        lines.append(category.upper())
        lines.extend(f"  {name}: {value}" for name, value in summary[category].items())
        lines.append("")
    lines.extend(summary["limitations"])
    return "\n".join(lines) + "\n"


def write_quality_summary(directory, *, expected_pairs=None):
    summary = summarize_quality(directory, expected_pairs=expected_pairs)
    directory = Path(directory)
    atomic_json(directory / "quality_summary.json", summary)
    temporary = directory / "quality_summary.txt.tmp"
    temporary.write_text(quality_text(summary), encoding="utf-8")
    temporary.replace(directory / "quality_summary.txt")
    return summary
