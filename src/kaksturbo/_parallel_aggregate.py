"""Serialize already-validated chunks in bounded processes, then merge in order."""
from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import uuid

from .checkpoint import atomic_json, contained, file_sha256, load_json
from .errors import ConfigurationError, EngineError, IntegrityError

FILES = ("result.tsv", "results.jsonl", "native_result.tsv", "native_results.jsonl",
         "pair_statistics.jsonl", "models.jsonl", "models.tsv", "qc.tsv", "data_dictionary.json")
HEADERS = frozenset(("result.tsv", "native_result.tsv", "models.tsv"))
SINGLETONS = frozenset(("qc.tsv", "data_dictionary.json"))


def build_fragment(api, job):
    """Reuse original complete validation/serialization; no scientific fitting."""
    output, destination = Path(job["output"]), Path(job["destination"])
    destination.mkdir(exist_ok=False)
    (destination / ".work").mkdir()
    started = time.perf_counter()
    names = api._aggregate_serial(output, job["state"], destination=destination)
    if tuple(names) != FILES:
        raise EngineError("Fragment serializer changed its complete output set")
    profile = dict(serialization_wall_seconds=time.perf_counter() - started,
                   scope="Single existing chunk: preserved validation and all nine public outputs; no fitting")
    return dict(schema="kaksturbo.aggregate_fragment/1", complete=True,
                chunk_index=job["state"]["chunks"][0]["index"], count=job["state"]["chunks"][0]["count"],
                files={name: {"sha256": file_sha256(destination / name), "bytes": (destination / name).stat().st_size}
                       for name in FILES}, profile=profile, destination=str(destination))


def allocation(state, environment=None):
    environment = os.environ if environment is None else environment
    text = environment.get("KAKSTURBO_AGGREGATE_WORKERS", "auto")
    cores = state["config"]["threads"]
    if text != "auto" and not re.fullmatch(r"[1-9][0-9]*", text):
        raise ConfigurationError("KAKSTURBO_AGGREGATE_WORKERS must be auto or a positive integer")
    if type(cores) is not int or cores < 1:
        raise ConfigurationError("Output processing requires a positive total logical-CPU budget")
    workers = min(8, cores, len(state["chunks"])) if text == "auto" else int(text)
    if not 1 <= workers <= min(8, cores, len(state["chunks"])):
        raise ConfigurationError("Output processes must not exceed 8, the total CPU budget, or the number of chunks")
    return workers


def windows_kernel():
    if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
        raise ConfigurationError("This output candidate requires 64-bit native Windows")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = C.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    kernel.GetProcessAffinityMask.restype = C.c_int
    kernel.SetProcessAffinityMask.argtypes = [C.c_void_p, C.c_size_t]
    kernel.SetProcessAffinityMask.restype = C.c_int
    kernel.GetLogicalProcessorInformationEx.argtypes = [C.c_int, C.c_void_p, C.POINTER(C.c_uint32)]
    kernel.GetLogicalProcessorInformationEx.restype = C.c_int
    return kernel


def process_mask(kernel):
    actual, system = C.c_size_t(), C.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), C.byref(actual), C.byref(system)):
        raise ConfigurationError("Cannot inspect Windows parent CPU affinity")
    if actual.value == 0 or actual.value & ~system.value:
        raise ConfigurationError("Invalid Windows process CPU affinity")
    return actual.value


def decode_core_masks(data, allowed_mask):
    """Decode group-0 physical cores; never assume adjacent SMT numbering."""
    cores, offset = [], 0
    while offset < len(data):
        if len(data) - offset < 32:
            raise ConfigurationError("Truncated Windows physical-core topology")
        relation, size = struct.unpack_from("<II", data, offset)
        groups = struct.unpack_from("<H", data, offset + 30)[0]
        if relation != 0 or groups < 1 or size < 32 + groups * 16 or offset + size > len(data):
            raise ConfigurationError("Malformed Windows physical-core topology")
        for index in range(groups):
            mask, group = struct.unpack_from("<QH", data, offset + 32 + index * 16)
            if group != 0:
                raise ConfigurationError("Parallel output currently supports Windows processor group 0 only")
            available = mask & allowed_mask
            if available:
                cores.append([bit for bit in range(64) if available & (1 << bit)])
        offset += size
    flattened = [cpu for core in cores for cpu in core]
    allowed = [bit for bit in range(64) if allowed_mask & (1 << bit)]
    if sorted(flattened) != allowed or len(flattened) != len(set(flattened)):
        raise ConfigurationError("Windows topology does not uniquely cover the allowed CPU set")
    return sorted(cores, key=lambda core: core[0])


def windows_resources():
    kernel = windows_kernel()
    allowed_mask = process_mask(kernel)
    size = C.c_uint32()
    kernel.GetLogicalProcessorInformationEx(0, None, C.byref(size))
    if size.value == 0:
        raise ConfigurationError("Cannot size Windows physical-core topology")
    buffer = C.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, C.byref(size)):
        raise ConfigurationError("Cannot read Windows physical-core topology")
    allowed = [bit for bit in range(64) if allowed_mask & (1 << bit)]
    return allowed, decode_core_masks(buffer.raw[:size.value], allowed_mask)


def verified_cpus(state, workers, *, automatic=False):
    allowed, cores = windows_resources()
    if len(allowed) != state["config"]["threads"]:
        raise ConfigurationError("Parent affinity width differs from its total core budget")
    if automatic:
        workers = min(workers, len(cores))
    if workers > len(cores) or workers < 1:
        raise ConfigurationError("Output workers must use distinct physical cores within the total budget")
    selected = [core[0] for core in cores[:workers]]
    return allowed, selected


def pin_output_cpu(cpu):
    if type(cpu) is not int or not 0 <= cpu < 64:
        raise ConfigurationError("Output worker requires one valid logical CPU")
    kernel = windows_kernel()
    inherited = process_mask(kernel)
    selected = 1 << cpu
    if not inherited & selected:
        raise ConfigurationError("Output worker CPU is outside its inherited parent budget")
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), selected) or process_mask(kernel) != selected:
        raise ConfigurationError("Output worker failed to acquire its assigned single CPU")
    return dict(inherited=[bit for bit in range(64) if inherited & (1 << bit)], actual=[cpu], pid=os.getpid())


def prepare(state, environment=None):
    """Choose output resources before fitting, without changing CPU affinity.

    Auto is opportunistic: an unsupported or unbound caller retains the original
    sequential output path. An explicit parallel request must already satisfy
    the complete quota; it is never silently shrunk or allowed to fail after fit.
    """
    if state["config"].get("mode") != "robust":
        return 1, None, None
    environment = os.environ if environment is None else environment
    requested = environment.get("KAKSTURBO_AGGREGATE_WORKERS", "auto")
    workers = allocation(state, environment)
    allowed = selected = None
    fallback = None
    if workers > 1:
        try:
            allowed, selected = verified_cpus(state, workers, automatic=requested == "auto")
            workers = len(selected)
        except ConfigurationError as error:
            if requested != "auto":
                raise ConfigurationError("Explicit parallel output configuration is invalid before fitting: " + str(error)) from error
            workers = 1
            fallback = str(error)
    if state.get("public_aggregation_workers", workers) != workers:
        raise IntegrityError("Resume must retain the original output-worker count")
    stored_cpus = state.get("public_aggregation_cpu_set")
    if stored_cpus is not None and stored_cpus != selected:
        raise IntegrityError("Resume must retain the original output CPU allocation")
    state["public_aggregation_workers"] = workers
    if selected is not None:
        state["public_aggregation_cpu_set"] = selected
    state["public_aggregation_selection"] = dict(
        requested=requested, workers=workers, parent_allowed_cpus=allowed,
        worker_cpus=selected, sequential_fallback_reason=fallback,
        policy="auto may use unchanged sequential output; explicit parallel resources are checked before fitting")
    return workers, allowed, selected


def verify_numerical_workers_finished(output, state):
    if state.get("status") != "aggregating" or not state["chunks"]:
        raise EngineError("Output processing may start only after all fitting has completed")
    for chunk in state["chunks"]:
        if chunk.get("status") != "complete" or not chunk.get("attempts") or chunk["attempts"][-1].get("status") != "complete":
            raise EngineError("An unfinished numerical chunk remains")
        execution = load_json(contained(output, chunk["attempts"][-1]["directory"]) / "engine_execution.json")
        owned = execution.get("owned_process", {})
        if execution.get("complete") is not True or owned.get("cleanup_PASS") is not True or owned.get("child_exitcode") != 0:
            raise EngineError("Numerical worker exit and owned-process cleanup were not confirmed")


def run_fragment(api, job, job_path, logs, timeout):
    # Each owned child gets one disjoint CPU, private environment and outputs.
    # No coordinator thread mutates os.environ or process affinity.
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("_aggregate_child.py")), str(job_path)]
    engine = api._engine()
    owned = engine._windows_owned(command, logs, engine.resource_root(), environment, timeout)
    if owned.get("status") != "completed" or owned.get("child_exitcode") != 0 or owned.get("cleanup_PASS") is not True:
        raise EngineError("Output fragment failed; inspect its retained owned-process logs")
    result = load_json(Path(job["destination"]) / "completed.json")
    if result.get("complete") is not True or result.get("chunk_index") != job["state"]["chunks"][0]["index"]:
        raise EngineError("Output fragment omitted its complete index-matched result")
    if result.get("CPU_affinity", {}).get("actual") != [job["cpu"]]:
        raise EngineError("Output fragment ran outside its allocated single CPU")
    return result


def merge_fragments(output, state, staging, fragments):
    """Hash-checked byte merge; deterministic original order and one TSV header."""
    from .results import write_data_dictionary
    expected_indexes = [chunk["index"] for chunk in state["chunks"]]
    if [item["chunk_index"] for item in fragments] != expected_indexes:
        raise EngineError("Output fragments are missing or out of original input order")
    if [item["count"] for item in fragments] != [chunk["count"] for chunk in state["chunks"]]:
        raise EngineError("Output fragments changed pair coverage")
    staging.mkdir(exist_ok=False)
    for name in FILES:
        if name in SINGLETONS:
            # Every serializer emitted these standard files. Require all copies
            # to agree; the original QC and data dictionary are checked below.
            hashes = {fragment["files"][name]["sha256"] for fragment in fragments}
            if len(hashes) != 1:
                raise IntegrityError("Output fragments disagree on a singleton artifact")
        expected_header = None
        with (staging / name).open("wb") as destination:
            for ordinal, fragment in enumerate(fragments):
                path = Path(fragment["destination"]) / name
                digest = hashlib.sha256()
                bytes_read = 0
                with path.open("rb") as source:
                    if name in HEADERS:
                        header = source.readline()
                        if not header.endswith(b"\n"):
                            raise IntegrityError("Missing or malformed fragment TSV header")
                        digest.update(header)
                        bytes_read += len(header)
                        if expected_header is None:
                            expected_header = header
                            destination.write(header)
                        elif header != expected_header:
                            raise EngineError("Result schemas differ across chunks")
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        bytes_read += len(block)
                        if name not in SINGLETONS or ordinal == 0:
                            destination.write(block)
                expected = fragment["files"][name]
                if bytes_read != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise IntegrityError("Output fragment changed before deterministic merging")
            destination.flush()
            os.fsync(destination.fileno())
    if file_sha256(staging / "qc.tsv") != file_sha256(contained(output, state["preflight"]["qc_path"])):
        raise IntegrityError("Merged QC differs from its original validated input")
    dictionary = staging / "checked_dictionary.json"
    write_data_dictionary(dictionary)
    if file_sha256(dictionary) != file_sha256(staging / "data_dictionary.json"):
        raise IntegrityError("Merged data dictionary differs from the original schema")
    # No standard final artifact is published until all fragments and schemas
    # have passed every check. A failed merge leaves staging and all sources.
    for name in FILES:
        os.replace(staging / name, output / name)


def aggregate(api, output, state):
    if state["config"].get("mode") != "robust":
        return api._aggregate_serial(output, state)
    workers, allowed, selected = prepare(state)
    if workers == 1:
        return api._aggregate_serial(output, state)
    verify_numerical_workers_finished(output, state)
    atomic_json(output / "checkpoint.json", state)
    started = time.perf_counter()
    # Leave room for the existing atomic JSON suffix on native Windows.
    attempt = output / ".work" / ("pa" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    profile = dict(schema="kaksturbo.public_aggregation_timing/2", workers=workers,
                   parent_allowed_cpus=allowed, worker_cpus=selected, chunks=[],
                   scope="Pure public-output workers after all numerical children exited; child times overlap; complete CLI timer unchanged")
    active, completed, failures = {}, {}, []
    pending = iter(state["chunks"])
    api._engine()  # Load the existing owned-process helper before threads start.

    def dispatch(pool, slot):
        try:
            chunk = next(pending)
        except StopIteration:
            return
        directory = attempt / f"f{chunk['index']:06d}"
        job = dict(output=str(output), destination=str(directory), cpu=selected[slot],
                   state={"config": state["config"], "preflight": state["preflight"], "chunks": [chunk]})
        job_path = attempt / f"job_{chunk['index']:06d}.json"
        atomic_json(job_path, job)
        future = pool.submit(run_fragment, api, job, job_path, attempt / f"logs_{chunk['index']:06d}",
                             state["config"].get("max_seconds", 1800.0))
        active[future] = (slot, chunk["index"])

    def collect(future):
        slot, index = active.pop(future)
        try:
            result = future.result()
        except BaseException as error:
            failures.append(dict(index=index, error_type=type(error).__name__, error=str(error)))
        else:
            completed[index] = result
        atomic_json(attempt / "progress.json", dict(completed=sorted(completed), failures=failures))
        return slot

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kaks-output") as pool:
            try:
                for slot in range(workers):
                    dispatch(pool, slot)
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    slots = [collect(future) for future in sorted(done, key=lambda value: active[value][1])]
                    if not failures:
                        for slot in slots:
                            dispatch(pool, slot)
            except BaseException:
                for future in list(active):
                    collect(future)
                raise
        if failures:
            raise EngineError("Output serialization failed; valid numerical checkpoints and fragments were preserved",
                              details={"failures": failures, "attempt": str(attempt)})
        profile["parallel_serialize_wall_seconds"] = time.perf_counter() - started
        fragments = [completed[chunk["index"]] for chunk in state["chunks"]]
        profile["chunks"] = [{"index": fragment["chunk_index"], "CPU_affinity": fragment["CPU_affinity"],
                              "serialization_profile": fragment["profile"]} for fragment in fragments]
        tick = time.perf_counter()
        merge_fragments(output, state, attempt / "merge_staging", fragments)
        profile["ordered_merge_seconds"] = time.perf_counter() - tick
        profile["total_seconds"] = time.perf_counter() - started
        atomic_json(output / "public_aggregation_timing.json", profile)
        atomic_json(attempt / "completed.json", dict(complete=True, profile=profile))
        return list(FILES)
    except BaseException as error:
        atomic_json(attempt / "failure.json", dict(complete=False, error_type=type(error).__name__,
                                                    error=str(error), completed=sorted(completed), failures=failures))
        raise
