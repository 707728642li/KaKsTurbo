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
from ._paths import absolute_path
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
    configured = os.environ.get("KAKSTURBO_RUNTIME_ROOT") or os.environ.get("KAKSFORGE_RUNTIME_ROOT")
    if configured:
        return absolute_path(configured)
    source = absolute_path(__file__)
    # This module is also copied into the engine's isolated scripts directory.
    if source.name == "product_v1_engine_runtime.py":
        return source.parents[1]
    from ._version import DEVELOPMENT_ENGINE, ENGINE_MANIFEST_SHA256
    bundled = source.parent / "_engine_root"
    manifest = bundled / MANIFEST_NAME
    if manifest.is_file() and file_sha(manifest) == ENGINE_MANIFEST_SHA256:
        return bundled
    project = source.parents[2]
    if source.parent.parent.name == "src" and (project / "pyproject.toml").is_file():
        return project / DEVELOPMENT_ENGINE
    return bundled


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


def _verify_conda_assets(manifest, prefix=None):
    """Verify build-time pinned Conda providers without changing their files."""
    assets = manifest.get("conda_external_assets", {})
    if not isinstance(assets, dict):
        raise EngineIntegrityError("Invalid Conda dependency manifest")
    root = absolute_path(prefix or sys.prefix)
    for relative, record in assets.items():
        parts = Path(relative).parts
        if not parts or parts[0] != "lib" or not isinstance(record, dict):
            raise EngineIntegrityError("Invalid Conda dependency path: " + str(relative))
        try:
            path = _relative_asset(root, relative)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(path, flags), "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != record.get("bytes"):
                    raise EngineIntegrityError("Conda dependency size changed: " + relative)
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != record.get("sha256"):
                    raise EngineIntegrityError("Conda dependency missing or changed: " + relative)
        except OSError as error:
            raise EngineIntegrityError("Conda dependency missing or unsafe: " + relative) from error
    return len(assets)


def validate_manifest(root=None):
    root = absolute_path(root or resource_root())
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise EngineIntegrityError("Scientific runtime is not installed: " + str(manifest_path))
    raw = manifest_path.read_bytes()
    if __package__ == "kaksturbo":
        from ._version import ENGINE_MANIFEST_SHA256
        if hashlib.sha256(raw).hexdigest() != ENGINE_MANIFEST_SHA256:
            raise EngineIntegrityError("This interface requires its pinned Schur engine; use the matching release")
    manifest = json.loads(raw)
    if manifest.get("schema") not in {"kaksforge.engine_assets/1", "kaksturbo.engine_assets/1"}:
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
    _verify_conda_assets(manifest)
    return {"schema": "kaksturbo.engine_identity/1", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "asset_count": len(files), "platform": manifest.get("platform", platform.system()),
            "release": manifest.get("version", "1.0.0"), "science_contract": manifest.get("science_contract"),
            "PASS": True}


def asset_identity(config=None):
    # Paths are deliberately absent so a byte-identical installation can move.
    identity = validate_manifest()
    identity.update(model_order=list(MODELS), genetic_code=1, method="MA", precision="FP64",
                    mode=config_value(config, "mode", "robust"))
    if identity["mode"] == "robust":
        build = resource_root() / "build/cpu-gradient-v2" / ("build.json" if os.name == "nt" else "linux-build.json")
        if build.is_file():
            limit = json.loads(build.read_text(encoding="utf-8-sig")).get("worker_count_limit")
            if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                identity["cpu_worker_limit"] = limit
                if config_value(config, "threads", 16) > limit:
                    from .errors import ConfigurationError
                    raise ConfigurationError(f"Installed engine supports at most {limit} CPU workers; reduce --threads or use the matching expanded engine")
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
    requested = int(os.environ.get("KAKSTURBO_CPU_THREADS", "16"))
    thread_limit = 32 if os.name == "nt" else 128
    if not 1 <= requested <= thread_limit:
        raise ValueError(f"Product CPU affinity request must be in 1..{thread_limit}")
    if os.name != "nt":
        original = os.sched_getaffinity(0)
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
    from .devices import inventory
    return inventory()


def select_gpu(config, snapshot=None):
    from .devices import choose
    return choose(config, snapshot)


def doctor(config=None):
    from .devices import DeviceError, check_gpu_package, probe_driver
    answer = dict(schema="kaksturbo.doctor/2", python=sys.version, platform=platform.platform(),
                  runtime_root=str(resource_root()), Torch_imported="torch" in sys.modules,
                  GPU_initialized=False, CPU_available=False, GPU_available=False,
                  GPU_execution_verified=False, numerical_fit_performed=False)
    libraries=[]
    historical=resource_root()/"build/cpu-reference-cached"/("KaKs_Calculator.exe" if os.name=="nt" else "KaKs_Calculator")
    answer["available_modes"]=["robust"] + (["reproduce"] if historical.is_file() else [])
    answer["GPU_backend_packaged"]=False
    try:
        answer["engine_identity"] = validate_manifest()
        native_dir=resource_root()/"envs-runtime/windows/native"
        token=os.add_dll_directory(str(native_dir)) if os.name=="nt" and native_dir.is_dir() else None
        try:
            for name in ("cpu-backend-v1-reviewed/kaks_cpu_backend.dll", "cpu-gradient-v2/kaks_cpu_gradient.dll",
                         "gradient-optimizer-schur-v1/kaks_optimizer.dll", "shared-statistics-v1-checked/kaks_statistics.dll",
                         "native-adapter-checked/kaks_adapter.dll"):
                libraries.append(C.CDLL(str(native_asset("build/"+name))))
        finally:
            if token: token.close()
        answer["CPU_available"]=True
        answer["CPU_status"]="Native assets verified and libraries loaded; no fit performed."
    except Exception as error:
        answer["engine_error"]=str(error)
        answer["CPU_status"]="Unavailable; reinstall the matching package or report doctor --json output."
    try:
        check_gpu_package(resource_root());answer["GPU_backend_packaged"]=True
    except DeviceError as error:
        answer["GPU_package_diagnostic"]=error.as_dict()
    try:
        snapshot=gpu_inventory();answer["GPU_inventory"]=snapshot
        check_gpu_package(resource_root())
        selected,_=select_gpu(config or {},snapshot);answer["selected_GPU"]=selected
        answer["GPU_status"]="Inventory/admission passed; CUDA driver and kernels have not been executed."
        if config_value(config,"probe_gpu",False):
            answer["GPU_driver_probe"]=probe_driver(selected)
            answer["GPU_initialized"]=True
            answer["GPU_status"]="Driver initialized and device/architecture checked; no context, memory allocation, or fit."
        answer["GPU_available"]=answer["CPU_available"]
    except DeviceError as error:
        answer["GPU_error"]=str(error);answer["GPU_diagnostic"]=error.as_dict()
        answer["GPU_status"]=error.details["reason"]
    except Exception as error:
        answer["GPU_error"]=str(error);answer["GPU_status"]="unexpected_preflight_error"
    requested=config_value(config,"backend","auto");selected_backend="cpu" if requested=="auto" else requested
    answer.update(requested_backend=requested,selected_backend=selected_backend,
                  diagnostic_scope="Installation/device preflight only; it does not certify numerical convergence.",
                  auto_policy="auto selects CPU; explicit gpu requests never silently fall back")
    answer["PASS"]=answer["CPU_available"] and (selected_backend=="cpu" or answer["GPU_available"])
    return answer


def preflight_configuration(args, optimizer_path):
    identity = validate_manifest()
    expected = os.environ.get("KAKSTURBO_ENGINE_MANIFEST_SHA256")
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
    ticket = dict(schema="kaksturbo.product_engine_ticket/1", engine_identity=identity,
                  publication_validation_authorized=False, release_candidate=True,
                  historical_freeze_used_as_authorization=False)
    return ticket, science, execution


def post_run_check(ticket):
    actual = validate_manifest()
    if actual != ticket["engine_identity"]:
        raise EngineIntegrityError("Scientific runtime changed during the chunk")
    return {"PASS": True, "scope": "Product engine asset manifest checked before and after numerical execution",
            "engine_identity": actual}
