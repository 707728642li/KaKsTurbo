"""Locate and verify the private scientific engine without importing CUDA/Torch.

The asset manifest belongs to this product release; historical benchmark freezes
are evidence for provenance and are never used to authorize a product run.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
import csv
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import struct
import stat
import sys

MODELS = ("JC", "F81", "K2P", "HKY", "TNEF", "TN", "K3P", "K3PUF", "TIMEF", "TIM", "TVMEF", "TVM", "SYM", "GTR")
MANIFEST_NAME = "engine-assets.json"


class EngineIntegrityError(RuntimeError):
    """An installed scientific asset differs from its release identity."""


def config_value(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def config_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def resource_root():
    configured = os.environ.get("KAKSFORGE_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    source = Path(__file__).resolve()
    # This module is also copied into the engine's isolated scripts directory.
    if source.name == "product_v1_engine_runtime.py":
        return source.parents[1]
    return source.parent / "_engine_root"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_asset(root, relative, directories=None):
    path = root / relative
    if Path(relative).is_absolute() or Path(relative).drive or ".." in Path(relative).parts:
        raise EngineIntegrityError("Invalid relative asset path: " + relative)
    directories = {} if directories is None else directories
    parent = path.parent
    if parent not in directories:
        directories[parent] = parent.resolve()
    resolved = directories[parent] / path.name
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EngineIntegrityError("Asset escapes installed runtime: " + relative) from error
    if resolved.is_symlink():
        raise EngineIntegrityError("Scientific asset must be a regular file, not a symlink: " + relative)
    return resolved


def validate_manifest(root=None):
    root = Path(root or resource_root()).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise EngineIntegrityError("Scientific runtime is not installed: " + str(manifest_path))
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "kaksforge.engine_assets/1":
        raise EngineIntegrityError("Unsupported scientific runtime manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise EngineIntegrityError("Scientific runtime manifest contains no assets")
    # Resolve the root once. Linux openat descriptors pin each directory and
    # reject symlinks; every file is still read and SHA256 checked on every call.
    # This avoids thousands of redundant ancestor stats on WSL/DrvFS.
    descriptors = {}
    directories = {}
    use_openat = os.name == "posix" and hasattr(os, "O_NOFOLLOW")
    try:
        if use_openat:
            descriptors[()] = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for relative, record in files.items():
            parts = Path(relative).parts
            if not parts or Path(relative).is_absolute() or Path(relative).drive or ".." in parts:
                raise EngineIntegrityError("Invalid relative asset path: " + relative)
            expected = record if isinstance(record, str) else record.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise EngineIntegrityError("Missing SHA256 for asset: " + relative)
            try:
                if use_openat:
                    prefix = ()
                    for part in parts[:-1]:
                        next_prefix = prefix + (part,)
                        if next_prefix not in descriptors:
                            descriptors[next_prefix] = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                                                dir_fd=descriptors[prefix])
                        prefix = next_prefix
                    fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptors[prefix])
                    handle = os.fdopen(fd, "rb")
                else:
                    handle = _relative_asset(root, relative, directories).open("rb")
                with handle:
                    info = os.fstat(handle.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        raise EngineIntegrityError("Scientific asset is not a regular file: " + relative)
                    if isinstance(record, dict) and record.get("bytes") is not None and info.st_size != record["bytes"]:
                        raise EngineIntegrityError("Scientific asset size changed: " + relative)
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                    if digest.hexdigest() != expected:
                        raise EngineIntegrityError("Scientific asset missing or changed: " + relative)
            except OSError as error:
                raise EngineIntegrityError("Scientific asset missing or unsafe: " + relative) from error
    finally:
        for fd in descriptors.values():
            os.close(fd)
    return {"schema": "kaksforge.engine_identity/1", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "asset_count": len(files), "platform": manifest.get("platform", platform.system()),
            "release": manifest.get("version", "1.0.0"), "science_contract": manifest.get("science_contract"),
            "PASS": True}


def asset_identity(config=None):
    # Paths are deliberately absent so a byte-identical installation can move.
    identity = validate_manifest()
    identity.update(model_order=list(MODELS), genetic_code=1, method="MA", precision="FP64",
                    mode=config_value(config, "mode", "robust"))
    return identity


def native_asset(relative):
    path = resource_root() / relative
    if os.name != "nt" and path.suffix.lower() == ".dll":
        path = path.with_name("lib" + path.stem + ".so")
    if os.name != "nt" and path.suffix.lower() == ".exe":
        path = path.with_suffix("")
    if not path.is_file():
        raise FileNotFoundError("Installed native engine component is missing: " + str(path))
    return path


def platform_build_metadata(path):
    """Select native build evidence without overwriting Windows provenance."""
    filename = "build.json" if os.name == "nt" else "linux-build.json"
    metadata_path = Path(path).parent / filename
    if not metadata_path.is_file():
        raise EngineIntegrityError("Native platform build record is missing: " + str(metadata_path))
    return json.loads(metadata_path.read_text(encoding="utf-8-sig"))


@contextmanager
def process_affinity():
    """Use actual available CPU topology; small machines remain supported."""
    requested = int(os.environ.get("KAKSFORGE_CPU_THREADS", "16"))
    if not 1 <= requested <= 128:
        raise ValueError("Product CPU affinity request must be in 1..128")
    if os.name != "nt":
        original = os.sched_getaffinity(0)
        if len(original) < requested:
            raise ValueError("Requested workers exceed allowed CPUs")
        selected = set(sorted(original)[:requested])
        os.sched_setaffinity(0, selected)
        try:
            actual = os.sched_getaffinity(0)
            if actual != selected:
                raise RuntimeError("Linux CPU affinity differs from selected logical processors")
            yield dict(platform="Linux", selected_logical_processors=sorted(selected),
                       logical_processor_count=len(selected), requested_workers=requested,
                       physical_core_count=None, physical_topology_verified=False,
                       original_parent_logical_processors=sorted(original))
        finally:
            os.sched_setaffinity(0, original)
        return
    from benchmark_cpu import affinity, kernel32
    kernel = kernel32()
    handle = kernel.GetCurrentProcess()
    original, system = affinity(kernel, handle)
    size = C.c_uint32()
    kernel.GetLogicalProcessorInformationEx(0, None, C.byref(size))
    buffer = C.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, C.byref(size)):
        raise C.WinError(C.get_last_error())
    cores = []
    offset = 0
    while offset < size.value:
        relationship, entry_size = struct.unpack_from("<II", buffer.raw, offset)
        if entry_size < 32 or offset + entry_size > size.value:
            raise RuntimeError("Malformed Windows processor topology")
        groups = struct.unpack_from("<H", buffer.raw, offset + 30)[0]
        if relationship == 0:
            for group_index in range(groups):
                mask, group = struct.unpack_from("<QH", buffer.raw, offset + 32 + 16 * group_index)
                available = mask & original
                if group == 0 and available:
                    cores.append(dict(physical_core_index=len(cores), group=0, mask=available,
                                      logical_processors=[bit for bit in range(64) if available & (1 << bit)]))
        offset += entry_size
    selected = []
    for core in cores:
        selected.extend(core["logical_processors"][:max(0, requested - len(selected))])
        if len(selected) >= requested:
            break
    if not selected:
        raise RuntimeError("No available Windows group-0 logical processors")
    mask = sum(1 << bit for bit in selected)
    chosen = [core for core in cores if core["mask"] & mask]
    if not kernel.SetProcessAffinityMask(handle, mask):
        raise C.WinError(C.get_last_error())
    try:
        if affinity(kernel, handle)[0] != mask:
            raise RuntimeError("Product CPU affinity verification failed")
        yield dict(platform="Windows", all_physical_cores=cores, selected_physical_cores=chosen,
                   selected_mask=mask, selected_mask_hex=hex(mask), original_parent_mask=original,
                   system_mask=system, physical_core_count=len(chosen), logical_processor_count=len(selected),
                   selected_logical_processors=selected, requested_workers=requested,
                   physical_topology_verified=True)
    finally:
        if not kernel.SetProcessAffinityMask(handle, original):
            raise C.WinError(C.get_last_error())


def gpu_inventory():
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu"
    query = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=20)
    processes = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=20)
    cards = []
    for row in csv.reader(query.stdout.splitlines()):
        if not row:
            continue
        parts = [item.strip() for item in row]
        cards.append(dict(index=int(parts[0]), uuid=parts[1], name=parts[2], total_MiB=int(parts[3]),
                          used_MiB=int(parts[4]), utilization_percent=int(parts[5])))
    if not cards:
        raise RuntimeError("nvidia-smi did not report any GPU")
    return {"cards": cards, "query_csv": query.stdout, "process_snapshot": processes.stdout}


def select_gpu(config, snapshot=None):
    snapshot = snapshot or gpu_inventory()
    cards = snapshot["cards"]
    # Measured batch512/cache256 peak is about 4.5 GiB, plus context and 2 GiB
    # independent reserve. This admission estimate never replaces runtime checks.
    minimum_free_mib = 8192
    eligible = [card for card in cards if card["total_MiB"] - card["used_MiB"] >= minimum_free_mib]
    selector = config_value(config, "device", "auto")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selectors = [item.strip() for item in visible.split(",")]
        permitted = []
        for item in selectors:
            matches = [card for card in cards if item == str(card["index"]) or (item.startswith("GPU-") and card["uuid"].startswith(item))]
            if len(matches) != 1:
                raise ValueError("CUDA_VISIBLE_DEVICES must resolve to unique physical GPUs")
            permitted.append(matches[0]["uuid"])
        if len(permitted) != len(set(permitted)):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate devices")
        eligible = [card for card in eligible if card["uuid"] in permitted]
    if selector != "auto":
        if selector.startswith("cuda:"):
            requested = int(selector.split(":", 1)[1])
            matches = [card for card in eligible if card["index"] == requested]
        elif selector.startswith("GPU-"):
            matches = [card for card in eligible if card["uuid"].startswith(selector)]
        else:
            raise ValueError("GPU selector must be auto, cuda:<physical-index>, or GPU UUID")
        if len(matches) != 1:
            raise MemoryError("Selected GPU is unavailable or lacks 8 GiB free admission headroom")
        selected = matches[0]
    else:
        idle = [card for card in eligible if card["utilization_percent"] == 0 and card["used_MiB"] <= 100]
        if idle:
            selected = next((card for card in idle if card["index"] == 1), idle[0])
        else:
            spare = [card for card in eligible if card["utilization_percent"] < 80]
            if not spare:
                raise RuntimeError("No GPU currently has memory and compute headroom; choose CPU or retry later")
            selected = min(spare, key=lambda card: (card["utilization_percent"], card["used_MiB"]))
    return dict(selected, admission_free_MiB=selected["total_MiB"] - selected["used_MiB"],
                admission_minimum_free_MiB=minimum_free_mib, runtime_reserve_bytes=2 * 1024 ** 3), snapshot


def doctor(config=None):
    answer = {"schema": "kaksforge.doctor/1", "python": sys.version, "platform": platform.platform(),
              "runtime_root": str(resource_root()), "Torch_imported": "torch" in sys.modules,
              "GPU_initialized": False, "CPU_available": False, "GPU_available": False}
    try:
        answer["engine_identity"] = validate_manifest()
        answer["CPU_available"] = True
    except Exception as error:
        answer["engine_error"] = str(error)
    try:
        snapshot = gpu_inventory()
        answer["GPU_inventory"] = snapshot
        if answer["CPU_available"]:
            select_gpu(config or {}, snapshot)
            answer["GPU_available"] = True
    except Exception as error:
        answer["GPU_error"] = str(error)
    answer["PASS"] = answer["CPU_available"]
    return answer


def preflight_configuration(args, optimizer_path):
    identity = validate_manifest()
    expected = os.environ.get("KAKSFORGE_ENGINE_MANIFEST_SHA256")
    if expected and identity["manifest_sha256"] != expected:
        raise EngineIntegrityError("Child engine identity differs from the parent checkpoint")
    metadata = platform_build_metadata(optimizer_path)
    science = dict(model_contract="pure-rates-strict-positive-v1", genetic_code=1, candidate_models=list(MODELS),
                   frequency_policy="historical-conditional-empirical-or-F3x4-v1", support_policy="StrictPositive",
                   derivatives="analytic", optimizer_contract=metadata["policy"],
                   optimizer_binary_sha256=file_sha(optimizer_path), time_bound_policy="fixed",
                   initial_time=.05, initial_rate=1.2, initial_omega=.4, max_iterations=args.max_iterations,
                   quality_tolerance=1e-6, zero_substitution_policy="unidentifiable-zero-observed-SNP-no-fitted-ratio-v1",
                   objective_absolute_tolerance=1e-10, objective_relative_tolerance=1e-12,
                   returned_gradient_policy="fresh-conditional-F-log-gradient-diagnostic-v1")
    execution = dict(backend=args.backend, workers=args.workers if args.backend == "cpu" else 1,
                     batch_tasks=args.batch_tasks, request_capacity=args.request_capacity,
                     max_seconds=args.max_seconds, complete_ma=True, device_selector=args.device)
    ticket = dict(schema="kaksforge.product_engine_ticket/1", engine_identity=identity,
                  publication_validation_authorized=False, release_candidate=True,
                  historical_freeze_used_as_authorization=False)
    return ticket, science, execution


def post_run_check(ticket):
    actual = validate_manifest()
    if actual != ticket["engine_identity"]:
        raise EngineIntegrityError("Scientific runtime changed during the chunk")
    return {"PASS": True, "scope": "Product engine asset manifest checked before and after numerical execution",
            "engine_identity": actual}
