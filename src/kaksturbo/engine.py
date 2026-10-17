"""Isolated, manifest-pinned numerical chunks for the public API.

No numerical dependency is imported into the user's interpreter. Each chunk
owns a fresh worker process, and CPU and CUDA retain separate explicit modes.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from ._paths import absolute_path
import signal
import subprocess
import sys
import time

from .runtime import (asset_identity, config_value, doctor, file_sha, native_asset,
                      resource_root, select_gpu)


class EngineError(RuntimeError):
    pass


engine_identity = asset_identity


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _windows_owned(command, logs, root, env, timeout):
    source = root / "scripts/product_v1_owned_job.py"
    spec = importlib.util.spec_from_file_location("_kaksturbo_owned_job", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if command[0] == sys.executable:
        index = 1
        while index < len(command) and command[index] in ('-B', '-I'):
            index += 1
        if index < len(command) and command[index].endswith('.py'):
            # Python's normal script bootstrap strips the extended prefix from
            # sys.path[0]. Preserve it for sibling imports in a deep install.
            bootstrap = ('import sys,runpy; from pathlib import Path; '
                         'p=sys.argv[1]; sys.argv=sys.argv[1:]; '
                         'sys.path.insert(0,str(Path(p).parent)); '
                         'runpy.run_path(p,run_name="__main__")')
            command = command[:index] + ['-c', bootstrap, str(absolute_path(command[index]))] + command[index + 1:]
    # Every engine input/output/resource argument is absolute. Win32 still
    # limits CreateProcess's current-directory field, even for extended paths;
    # a volume root is a read-only working directory, not a data staging area.
    return module.run_owned_command(command, output=logs, cwd=Path(root.anchor), env=env, timeout=timeout, cleanup_timeout=15.)


def _posix_owned(command, logs, root, env, timeout):
    logs.mkdir(parents=True)
    started = time.perf_counter()
    def send_group(pid, sig):
        try:
            os.killpg(pid, sig)
            return True
        except ProcessLookupError:
            return False

    def group_exists(pid):
        return send_group(pid, 0)
    with (logs / "stdout.log").open("wb") as stdout, (logs / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        status = "completed"
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
            send_group(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                send_group(process.pid, signal.SIGKILL)
                code = process.wait(timeout=15)
        except BaseException:
            send_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                send_group(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            raise
    group_empty = not group_exists(process.pid)
    # This process group was created exclusively for this chunk. A surviving
    # descendant is a failed cleanup, even when its parent exited successfully.
    if not group_empty:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            send_group(process.pid, sig)
            deadline = time.monotonic() + 15.
            while time.monotonic() < deadline:
                group_empty = not group_exists(process.pid)
                if group_empty:
                    break
                time.sleep(.05)
            if group_empty:
                break
    report = dict(schema="kaksturbo.owned_process/1", status=status, child_exitcode=code,
                  cleanup_PASS=group_empty, owned_process_group_empty=group_empty,
                  pid=process.pid, full_process_wall_seconds=time.perf_counter() - started,
                  command=command, owns_new_process_group=True)
    _write_json(logs / "report.json", report)
    return report


def run_chunk(input_path: Path, output_dir: Path, config):
    """Run one validated AXT chunk of at most 1000 pairs; preserve all outputs.

    ``auto`` selects CPU in release 1.0. Explicit CUDA executes one selected
    physical GPU with no silent runtime fallback or change of estimator.
    """
    started = time.perf_counter()
    identity = asset_identity(config)
    root = resource_root()
    input_path = absolute_path(input_path)
    output_dir = absolute_path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Preserve an existing engine chunk: " + str(output_dir))
    mode = config_value(config, "mode", "robust")
    requested_backend = config_value(config, "backend", "auto")
    if mode not in ("robust", "reproduce") or requested_backend not in ("auto", "cpu", "gpu"):
        raise ValueError("Unknown scientific mode or execution backend")
    backend = "cpu" if requested_backend == "auto" else requested_backend
    if mode == "reproduce" and backend != "cpu":
        raise ValueError("Historical reproduce mode currently requires the CPU estimator")
    if mode == "reproduce":
        try:
            native_asset("build/cpu-reference-cached/KaKs_Calculator.exe")
        except FileNotFoundError as error:
            from .errors import ConfigurationError
            raise ConfigurationError("This package does not contain the historical reproduce-mode executable. Use --mode robust, or install a package that explicitly supports reproduce mode.") from error
    if config_value(config, "method", "MA") != "MA" or config_value(config, "genetic_code", 1) != 1:
        raise ValueError("KaKsTurbo 1.0 supports MA, all 14 models, and standard genetic code 1")
    threads = config_value(config, "threads", 16)
    iterations = config_value(config, "max_iterations", 500)
    timeout = config_value(config, "max_seconds", 600.)
    batch = config_value(config, "batch_tasks", 512)
    cache = config_value(config, "gpu_cache_buckets", 256)
    capacity = config_value(config, "request_capacity", 16384)
    thread_limit = 32 if os.name == "nt" else 128
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= thread_limit:
        raise ValueError(f"CPU threads must be an integer in 1..{thread_limit}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Chunk deadline must be finite and positive")
    if iterations != 500:
        raise ValueError("Release 1.0 retains the validated 500-iteration scientific contract")
    if not isinstance(batch, int) or not 32 <= batch <= 512 or batch & (batch - 1) or not isinstance(cache, int) or not 1 <= cache <= 256 or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("GPU batch/cache and request capacity are outside the bounded release configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    numerical_dir = output_dir / "engine"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               KAKSFORGE_RUNTIME_ROOT=str(root), KAKSFORGE_ENGINE_MANIFEST_SHA256=identity["manifest_sha256"],
               KAKSFORGE_CPU_THREADS=str(threads))
    selected = None
    if backend == "gpu":
        from .devices import check_gpu_package, probe_driver
        check_gpu_package(root)
        selected, inventory = select_gpu(config)
        inventory["driver_probe"] = probe_driver(selected)
        _write_json(output_dir / "GPU_admission.json", dict(selected=selected, inventory=inventory))
        env["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    entry = root / "scripts" / ("product_v1_engine_fit.py" if mode == "robust" else "product_v1_engine_reproduce.py")
    if os.name == 'nt' and mode == 'reproduce':
        entry = absolute_path(__file__).with_name('_windows_reproduce.py')
    command = [sys.executable, "-B", str(entry), "--input", str(input_path), "--output", str(numerical_dir),
               "--workers", str(threads), "--max-seconds", str(timeout)]
    if mode == "robust":
        command.extend(["--backend", backend, "--batch-tasks", str(batch), "--gpu-cache-buckets", str(cache),
                        "--request-capacity", str(capacity), "--max-iterations", str(iterations),
                        "--complete-ma", "--derivatives", "analytic", "--time-bound", "fixed",
                        "--stream-count", "1", "--schedule", "stable_keys", "--observed-layout", "original"])
        if selected:
            command.extend(["--device", "cuda:" + str(selected["index"])])
    input_before = file_sha(input_path)
    executor = _windows_owned if os.name == "nt" else _posix_owned
    owned = executor(command, output_dir / "owned_process", root, env, timeout + 120.)
    execution = dict(schema="kaksturbo.engine_execution/1", mode=mode, requested_backend=requested_backend,
                     backend=backend, auto_policy="CPU selected by release 1.0 conservative policy" if requested_backend == "auto" else None,
                     engine_identity=identity, input_sha256=input_before, owned_process=owned,
                     selected_GPU=selected, complete=False)
    _write_json(output_dir / "engine_execution.json", execution)
    if owned.get("status") != "completed" or owned.get("child_exitcode") != 0 or owned.get("cleanup_PASS") is not True:
        raise EngineError("Scientific chunk failed; inspect " + str(output_dir / "owned_process/stderr.log"))
    paths = {"models": numerical_dir / "models.json", "pair_statistics": numerical_dir / "pair_statistics.json",
             "result": numerical_dir / "result.tsv", "run_manifest": numerical_dir / "run_manifest.json"}
    if not all(path.is_file() for path in paths.values()):
        raise EngineError("Scientific engine omitted required output files")
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    quality = manifest.get("fit_quality_passed", manifest.get("quality_passed"))
    if manifest.get("complete") is not True or quality is not True:
        raise EngineError("Numerical engine did not pass the complete fitting-quality gate")
    if file_sha(input_path) != input_before or asset_identity(config) != identity:
        raise EngineError("Input or scientific runtime changed during the chunk")
    if selected and manifest.get("selected_GPU", {}).get("uuid") != selected["uuid"]:
        raise EngineError("Engine used a different physical GPU from admission")
    execution.update(complete=True, quality_passed=True, count=manifest["count"],
                     full_process_wall_seconds=time.perf_counter() - started,
                     outputs={name: dict(path=str(path), sha256=file_sha(path)) for name, path in paths.items()})
    _write_json(output_dir / "engine_execution.json", execution)
    return {**{name: str(path) for name, path in paths.items()}, "mode": mode, "backend": backend,
            "count": manifest["count"], "quality": {"passed": True, "fit_quality_passed": quality},
            "engine_identity": identity, "execution_manifest": str(output_dir / "engine_execution.json"),
            "full_process_wall_seconds": execution["full_process_wall_seconds"]}
