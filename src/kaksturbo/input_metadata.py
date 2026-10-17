"""Bind prepared metadata to the exact ordered sequence content, never infer pairs."""
from dataclasses import replace
import json
from pathlib import Path
from ._paths import absolute_path
from .checkpoint import file_sha256
from .errors import InputError


def resolve_input(input_path, config):
    path = absolute_path(input_path)
    directory_input = path.is_dir()
    if directory_input:
        path = path / "pairs.axt"
    manifest = path.parent / "preparation.json"
    if directory_input or (path.name == "pairs.axt" and manifest.is_file()):
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info.get("schema") not in {"kaksturbo.input_preparation/1", "kaksforge.input_preparation/1"} or info.get("complete") is not True:
                raise ValueError("incomplete preparation manifest")
            metadata = path.parent / "pairs.metadata.jsonl"
            for item in (path, metadata):
                if file_sha256(item) != info["sha256"][item.name]:
                    raise ValueError("prepared file hash mismatch: " + item.name)
            if config.input_format != "axt" or config.pairs_manifest:
                raise ValueError("prepared directories require AXT configuration")
            if config.input_metadata and Path(config.input_metadata) != metadata:
                raise ValueError("explicit metadata conflicts with preparation.json")
            config = replace(config, input_metadata=str(metadata))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InputError("Invalid prepared input: " + str(exc)) from exc
    return path, config


def attach_metadata(records, metadata_path, report_error):
    if not metadata_path:
        yield from records
        return
    from .io import pair_identity
    try:
        with Path(metadata_path).open(encoding="utf-8") as stream:
            for index, (pair, metadata) in enumerate(records):
                try:
                    saved = json.loads(next(stream))
                except StopIteration as exc:
                    raise ValueError("metadata has fewer rows than the input") from exc
                if (not isinstance(saved, dict) or type(saved.get("pair_index")) is not int
                        or saved["pair_index"] != index or saved.get("pair_id") != pair.pair_id
                        or saved.get("stable_pair_sha256") != pair_identity(pair)):
                    raise ValueError(f"metadata identity/order mismatch at pair {index + 1}: {pair.pair_id}")
                linked = dict(metadata)
                for field in ("seq1_id", "seq2_id"):
                    value = saved.get(field)
                    if value is not None and (not isinstance(value, str) or not value or any(c in value for c in "\t\r\n")):
                        raise ValueError(f"invalid {field} at pair {index + 1}")
                    linked[field] = value
                # Keep the actual AXT error location and QC; provenance is separate.
                linked["prepared_source_location"] = saved.get("source_location")
                linked["metadata_source"] = str(metadata_path)
                yield pair, linked
            if stream.read().strip():
                raise ValueError("metadata has more rows than the input")
    except (OSError, ValueError, TypeError) as exc:
        report_error(InputError("Prepared metadata validation failed: " + str(exc),
                               details={"metadata_path": str(metadata_path)}))
