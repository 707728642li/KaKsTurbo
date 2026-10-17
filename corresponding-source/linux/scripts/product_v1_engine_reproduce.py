"""Historical MA estimator with explicit unavailable candidate diagnostics.

The unmodified cached historical executable owns all 14 candidate fits. Its
22-column output is preserved; no unexposed per-model states are fabricated.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_cpu import ROOT, SCHEMA, dump, file_sha, schedule_shards, tsv_bytes
from kakscuda import MODEL_ORDER, check_rows, parse_input
from product_v1_engine_runtime import native_asset, post_run_check, process_affinity, validate_manifest


def run(args):
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Preserve an earlier scientific chunk")
    identity = validate_manifest()
    expected = os.environ.get("KAKSFORGE_ENGINE_MANIFEST_SHA256")
    if expected and identity["manifest_sha256"] != expected:
        raise RuntimeError("Historical child engine identity differs from parent")
    raw = args.input.read_bytes()
    input_sha = file_sha(args.input)
    records = parse_input(raw)
    if not 1 <= len(records) <= 1000:
        raise ValueError("Historical engine accepts chunks of 1..1000 pairs")
    executable = native_asset("build/cpu-reference-cached/KaKs_Calculator.exe")
    executable_sha = file_sha(executable)
    requested = [record["pair_id"] for record in records]
    by_name = {record["pair_id"]: record for record in records}
    args.output.mkdir(parents=True)
    jobs_directory = args.output / "historical_jobs"
    jobs_directory.mkdir()
    jobs = []
    for index, names in enumerate(schedule_shards(requested, by_name, "dynamic", 64)):
        if not names:
            continue
        source = jobs_directory / ("chunk_%04d.axt" % index)
        target = jobs_directory / ("chunk_%04d.tsv" % index)
        log = jobs_directory / ("chunk_%04d.log" % index)
        source.write_bytes(b"".join(by_name[name]["record"] for name in names))
        jobs.append((index, names, source, target, log))
    started = time.perf_counter()
    deadline = started + args.max_seconds
    with process_affinity() as topology:
        def worker(job):
            index, names, source, target, log = job
            tick = time.perf_counter()
            remaining = deadline - tick
            if remaining <= 0:
                raise TimeoutError("Historical chunk deadline exceeded")
            with log.open("wb") as stream:
                process = subprocess.Popen([str(executable), "-i", str(source), "-o", str(target), "-m", "MA", "-c", "1"],
                                           stdout=stream, stderr=subprocess.STDOUT)
                try:
                    if os.name == "nt":
                        from benchmark_cpu import affinity, kernel32
                        inherited, _ = affinity(kernel32(), int(process._handle))
                        if inherited != topology["selected_mask"]:
                            raise RuntimeError("Historical worker inherited unexpected CPU affinity")
                        actual_affinity = hex(inherited)
                    else:
                        actual_affinity = sorted(os.sched_getaffinity(process.pid))
                    code = process.wait(timeout=remaining)
                except BaseException:
                    # This worker only owns the process it just created.
                    process.kill()
                    process.wait()
                    raise
            if code:
                raise RuntimeError("Historical CLI failed: " + str(log))
            return dict(chunk=index, pairs=len(names), pid=process.pid, seconds=time.perf_counter() - tick,
                        log=str(log), output=str(target), actual_affinity=actual_affinity,
                        inherited_parent_affinity_verified=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            completed_jobs = list(pool.map(worker, jobs))
    rows = []
    for _, _, _, target, _ in jobs:
        # Original CLI writes raw tab-separated identifiers without CSV quoting.
        # QUOTE_NONE preserves legal printable identifiers beginning with '"'.
        with target.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            if reader.fieldnames != SCHEMA:
                raise RuntimeError("Historical CLI output has an unexpected 22-column header")
            candidate_rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in candidate_rows):
            raise RuntimeError("Historical CLI emitted a malformed tab-separated row")
        rows.extend(candidate_rows)
    check_rows(rows, requested)
    order = {name: index for index, name in enumerate(requested)}
    rows.sort(key=lambda row: order[row["Sequence"]])
    (args.output / "result.tsv").write_bytes(tsv_bytes(rows, SCHEMA))
    model_rows = []
    summaries = []
    for index, row in enumerate(rows):
        for model_index, model in enumerate(MODEL_ORDER):
            model_rows.append(dict(slot=index * 14 + model_index, pair=row["Sequence"], model=model,
                                   status="historical_candidate_not_exported", result18=None,
                                   optimizer_executed=True, diagnostics_available=False,
                                   quality_passed=None,
                                   explanation="The historical estimator computes all 14 candidates internally; its public CLI does not export their fitted states."))
        summaries.append(dict(pair_id=row["Sequence"], mode="reproduce", historical_result=row,
                              candidate_diagnostics_available=False,
                              MA=dict(status="unavailable" if row["Ka/Ks"] == "NA" else "available")))
    dump(args.output / "models.json", model_rows)
    dump(args.output / "pair_statistics.json", summaries)
    if file_sha(args.input) != input_sha or file_sha(executable) != executable_sha:
        raise RuntimeError("Historical input/executable changed during execution")
    assets = post_run_check(dict(engine_identity=identity))
    manifest = dict(schema="kaksforge.product_reproduce_chunk/1", complete=True, mode="reproduce", backend="cpu",
                    count=len(records), requested_pairs=requested, method="MA", genetic_code=1,
                    candidate_models=list(MODEL_ORDER), candidate_diagnostics_available=False,
                    input=str(args.input), input_sha256=input_sha, CPU_executable=str(executable),
                    CPU_executable_sha256=executable_sha, CPU_workers=args.workers, CPU_jobs=completed_jobs,
                    CPU_topology=topology, result_columns=SCHEMA, historical_result_22_columns=True,
                    fit_quality_passed=True, quality_passed=True,
                    quality_scope="Complete original CLI rows, exact identifiers, all 22 output columns and finite-or-NA values; historical estimator has no robust finite-coordinate certification.",
                    historical_optimizer_quality_certified=False,
                    scientific_algorithm="Original code-1 MA/GY94/QL/bounded BFGS with exact 64-entry lfun cache",
                    product_engine_asset_identity=identity, asset_post_run_audit=assets,
                    historical_freeze_used_as_authorization=False, wall_seconds=time.perf_counter() - started)
    dump(args.output / "run_manifest.json", manifest)
    print(json.dumps(dict(complete=True, quality_passed=True, count=len(records))))
    return 0


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=600.)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not args.max_seconds > 0:
        raise ValueError("Invalid historical chunk resource configuration")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
