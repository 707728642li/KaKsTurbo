"""FP64 batched GY94 same-x matrix route; no optimizer or fitted MA output.

prepare never imports torch. evaluate imports torch lazily; CUDA is used only
with explicit --device cuda:N, after nvidia-smi. Frozen input is read once and
all input tensors remain resident across repeated evaluations.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_cpu import axt_records, affinity, dump, kernel32, table, topology
from native_adapter import NativeAdapter

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["JC", "F81", "K2P", "HKY", "TNEF", "TN", "K3P", "K3PUF",
          "TIMEF", "TIM", "TVMEF", "TVM", "SYM", "GTR"]
NKAPPA = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5]
# Exactly code-1 transl_table[0] in immutable base.cpp, T/C/A/G order.
CODE1 = "FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
SENSE = [i for i, aa in enumerate(CODE1) if aa != "!"]
RATE_PAIRS = {(0, 1): 0, (2, 3): 1, (0, 2): 2,
              (1, 3): 3, (0, 3): 4, (1, 2): 5}
RATE_MAP = [[5, 5, 5, 5, 5, 5], [0, 0, 5, 5, 5, 5],
            [0, 1, 5, 5, 5, 5], [0, 0, 1, 1, 5, 5],
            [0, 1, 2, 2, 5, 5], [0, 0, 1, 2, 3, 5],
            [0, 1, 2, 3, 4, 5]]
POLICY = {"relative_tolerance": 1e-12, "absolute_tolerance": 1e-10,
          "description": "Declared primitive parity tolerance, all 96 FP64 values; unchanged from handwritten CUDA route."}


def codon_edges():
    """Same one-change edges in original i outer / j<i inner order."""
    edges = []
    for i, code_i in enumerate(SENSE):
        bases_i = [code_i // 16, code_i // 4 % 4, code_i % 4]
        for j in range(i):
            code_j = SENSE[j]
            bases_j = [code_j // 16, code_j // 4 % 4, code_j % 4]
            changes = [k for k in range(3) if bases_i[k] != bases_j[k]]
            if len(changes) == 1:
                k = changes[0]
                rate = RATE_PAIRS[tuple(sorted((bases_i[k], bases_j[k])))]
                edges.append((i, j, rate, CODE1[code_i] == CODE1[code_j]))
    return edges


def default_parameters(pair_count):
    result = []
    for _ in range(pair_count):
        for nkappa in NKAPPA:
            x = [.05, 1.2, 1.1, 1.3, 1.4, 1.5, 1.6]
            x[nkappa + 1] = .4
            result.append(x)
    return result


class PreparedPairs:
    """Original native preparation, held in RAM; no CPU candidate fitting."""
    def __init__(self, sample, count, adapter_path=None, pair_id=None):
        self.adapter = NativeAdapter(adapter_path or ROOT / "build/native-adapter-checked/kaks_adapter.dll")
        self.pairs, self.meta, self.values = [], [], []
        self.z0, self.z1, self.weights, self.gc = [], [], [], []
        self.raw_input = (sample / "benchmark.axt").read_bytes()
        self.input_sha256 = hashlib.sha256(self.raw_input).hexdigest()
        records = axt_records(self.raw_input)
        self.ids = [pair_id] if pair_id else [r["pair_id"] for r in table(sample / "pairs.tsv")[:count]]
        if not self.ids or (not pair_id and len(self.ids) != count):
            raise ValueError("Requested count exceeds frozen sample")
        try:
            for name in self.ids:
                lines = records[name].rstrip(b"\n").split(b"\n")[1:]
                middle = len(lines) // 2
                pair = self.adapter.prepare(name, b"".join(lines[:middle]), b"".join(lines[middle:]))
                self.pairs.append(pair)
                meta, values, z0, z1, weight, gc = pair.export(sum(map(len, self.weights)))
                self.meta.append(list(meta)); self.values.append(list(values))
                self.z0.append(list(z0)); self.z1.append(list(z1))
                self.weights.append(list(weight)); self.gc.append(list(gc))
        except BaseException:
            self.close()
            raise
        self.max_patterns = max(map(len, self.weights))
        self.edges = codon_edges()

    def manifest(self):
        return {"schema": "kakscuda.torch_matrix_prepare/1", "pair_count": len(self.ids),
                "requested_pairs": self.ids, "input_sha256": self.input_sha256,
                "adapter": str(self.adapter.path),
                "adapter_sha256": hashlib.sha256(self.adapter.path.read_bytes()).hexdigest(),
                "model_order": MODELS, "genetic_code": 1,
                "pattern_count": sum(map(len, self.weights)),
                "max_patterns_per_pair": self.max_patterns, "one_change_edges": len(self.edges),
                "CPU_candidate_fitting": False, "mode": "same-x single evaluation only",
                "policy": POLICY}

    def close(self):
        for pair in self.pairs:
            pair.close()
        self.pairs = []
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TorchMatrixEvaluator:
    """Batch Q/eigh or matrix exponential with resident input."""
    def __init__(self, prepared, device="cpu", batch_tasks=896, reduction="tree", solver="default",
                 likelihood_backend="eigh"):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.prepared, self.batch_tasks, self.reduction = prepared, batch_tasks, reduction
        if reduction not in ("tree", "ordered") or batch_tasks < 1:
            raise ValueError("Invalid reduction or batch size")
        if likelihood_backend not in ("eigh", "matrix-exp"):
            raise ValueError("Invalid likelihood backend")
        self.likelihood_backend = likelihood_backend
        if self.device.type == "cuda":
            if self.device.index is None:
                raise ValueError("Explicit CUDA device ordinal is required")
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.preferred_linalg_library(solver)
        elif self.device.type != "cpu":
            raise ValueError("Only CPU or one explicit CUDA device is supported")
        self.task_count = len(prepared.ids) * 14
        self.estimated_peak_bytes = (min(self.task_count, batch_tasks) * 61 * 61 * 8 * 16
                                     + len(prepared.ids) * prepared.max_patterns * 24
                                     + self.task_count * 96 * 8 + 512 * 1024**2)
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            if self.estimated_peak_bytes + 2 * 1024**3 > free:
                raise MemoryError("Estimated peak plus 2 GiB reserve exceeds current free GPU memory")
            self.memory_before = {"free_bytes": free, "total_bytes": total,
                                  "estimated_peak_bytes": self.estimated_peak_bytes,
                                  "reserve_bytes": 2 * 1024**3}
        else:
            self.memory_before = {"estimated_peak_bytes": self.estimated_peak_bytes}
        self._upload()

    def tensor(self, data, dtype=None):
        return self.torch.tensor(data, dtype=dtype or self.torch.float64, device=self.device)

    def _upload(self):
        t, p = self.torch, self.prepared
        self.model = self.tensor(list(range(14)) * len(p.ids), t.int64)
        self.nkappa = self.tensor(NKAPPA * len(p.ids), t.int64)
        self.rate_map = self.tensor([RATE_MAP[m // 2] for m in list(range(14)) * len(p.ids)], t.int64)
        self.active_parameter = t.arange(6, device=self.device)[None, :] < self.nkappa[:, None]
        self.length = self.tensor([v[0] for v in p.values]).repeat_interleave(14)
        self.snp = self.tensor([v[1] for v in p.values]).repeat_interleave(14)
        host_pi = []
        for v in p.values:
            host_pi.extend([[1. / 61] * 64 if m % 2 == 0 else v[2:66] for m in range(14)])
        self.pi = self.tensor(host_pi)
        width = p.max_patterns
        self.z0 = self.tensor([z + [0] * (width - len(z)) for z in p.z0], t.int64)
        self.z1 = self.tensor([z + [0] * (width - len(z)) for z in p.z1], t.int64)
        self.weight = self.tensor([w + [0.] * (width - len(w)) for w in p.weights])
        self.edge_i = self.tensor([e[0] for e in p.edges], t.int64)
        self.edge_j = self.tensor([e[1] for e in p.edges], t.int64)
        self.edge_rate = self.tensor([e[2] for e in p.edges], t.int64)
        self.edge_syn = self.tensor([e[3] for e in p.edges], t.bool)
        self.pattern_columns = t.arange(width, device=self.device)
        self.groups = {}
        for start in range(0, self.task_count, self.batch_tasks):
            grouped = defaultdict(list)
            for local in range(min(self.batch_tasks, self.task_count - start)):
                active = tuple(i for i, freq in enumerate(host_pi[start + local][:61]) if freq > 1e-6)
                if not active:
                    raise ValueError("Original eigenQREV has no active frequencies")
                grouped[active].append(local)
            self.groups[start] = [(self.tensor(jobs, t.int64), self.tensor(active, t.int64))
                                  for active, jobs in grouped.items()]
        self.synchronize()

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _sum_last(self, values):
        if self.reduction == "tree":
            return values.sum(dim=-1)
        total = self.torch.zeros(values.shape[:-1], dtype=values.dtype, device=self.device)
        for i in range(values.shape[-1]):
            total = total + values[..., i]
        return total

    def _parse_rates(self, rates, start, end):
        # parseSubRates is intentionally called three times overall. K3P and
        # TVM mappings are not idempotent; raw x must NOT be reloaded for stats.
        return rates.gather(1, self.rate_map[start:end])

    def _q_stats(self, rates, pi, omega):
        t = self.torch
        lower = rates[:, self.edge_rate] * pi[:, self.edge_j]
        upper = rates[:, self.edge_rate] * pi[:, self.edge_i]
        pij = (2. * pi[:, self.edge_i]) * lower
        zeros = t.zeros_like(pij)
        rs = self._sum_last(t.where(self.edge_syn[None, :], pij, zeros))
        ra0 = self._sum_last(t.where(self.edge_syn[None, :], zeros, pij))
        ra = self._sum_last(t.where(self.edge_syn[None, :], zeros, pij * omega[:, None]))
        mr = rs + ra
        multiplier = t.where(self.edge_syn[None, :], t.ones_like(pij), omega[:, None])
        q = t.zeros((len(pi), 61, 61), dtype=t.float64, device=self.device)
        q[:, self.edge_i, self.edge_j] = lower * multiplier
        q[:, self.edge_j, self.edge_i] = upper * multiplier
        q.diagonal(dim1=-2, dim2=-1).copy_(-self._sum_last(q))
        return q, rs, ra0, ra, mr

    def evaluate(self, parameters=None):
        t = self.torch
        host_x = default_parameters(len(self.prepared.ids)) if parameters is None else parameters
        x = host_x if isinstance(host_x, t.Tensor) else self.tensor(host_x)
        if x.shape != (self.task_count, 7) or x.dtype != t.float64 or x.device != self.device:
            raise ValueError("x must be resident float64 [pair_count*14,7] on the selected device")
        output = t.zeros((self.task_count, 96), dtype=t.float64, device=self.device)
        with t.inference_mode():
            for start in range(0, self.task_count, self.batch_tasks):
                end = min(self.task_count, start + self.batch_tasks)
                xb, pi = x[start:end], self.pi[start:end, :61]
                omega = xb.gather(1, (self.nkappa[start:end] + 1)[:, None])[:, 0]
                raw_rates = t.where(self.active_parameter[start:end], xb[:, 1:7], t.ones_like(xb[:, 1:7]))
                rates1 = self._parse_rates(raw_rates, start, end)
                q, _, _, _, mr = self._q_stats(rates1, pi, omega)
                pair_indices = t.arange(start, end, device=self.device) // 14
                z0, z1, weights = self.z0[pair_indices], self.z1[pair_indices], self.weight[pair_indices]
                lnL = t.zeros(end - start, dtype=t.float64, device=self.device)
                for jobs, active in self.groups[start]:
                    pg = pi[jobs][:, active]
                    sqrt_pi = pg.sqrt()
                    qg = q[jobs][:, active][:, :, active]
                    # Original lower triangle only, followed by exact mirror.
                    sg = (qg * sqrt_pi[:, :, None]) / sqrt_pi[:, None, :]
                    lower = t.tril(sg, diagonal=-1)
                    symmetric = lower + lower.transpose(-1, -2) + t.diag_embed(qg.diagonal(dim1=-2, dim2=-1))
                    if self.likelihood_backend == "matrix-exp":
                        # Preserve original lower-only S and pi>1e-6 active
                        # compression. This evaluates the same mathematical
                        # transition via a distinct FP64 numerical path.
                        scaled = (symmetric * xb[jobs, 0, None, None]) / mr[jobs, None, None]
                        exponential = t.linalg.matrix_exp(scaled)
                        transition = (exponential / sqrt_pi[:, :, None]) * sqrt_pi[:, None, :]
                    else:
                        roots, evec = t.linalg.eigh(symmetric, UPLO="L")
                        roots, evec = roots.flip(-1), evec.flip(-1)
                        roots = roots / mr[jobs, None]
                        ug = evec / sqrt_pi[:, :, None]
                        vg = evec.transpose(-1, -2) * sqrt_pi[:, None, :]
                        expt = (xb[jobs, 0, None] * roots).exp()
                        if self.reduction == "tree":
                            transition = (ug * expt[:, None, :]) @ vg
                        else:
                            transition = t.zeros_like(ug)
                            for k in range(len(active)):
                                transition = transition + ((ug[:, :, k] * expt[:, k, None])[:, :, None] * vg[:, k, None, :])
                    # Source embeds inactive states as identity U/V and Root=0.
                    full = t.eye(61, dtype=t.float64, device=self.device).expand(len(jobs), 61, 61).clone()
                    full[:, active[:, None], active[None, :]] = transition
                    zg0, zg1 = z0[jobs], z1[jobs]
                    fh = full[t.arange(len(jobs), device=self.device)[:, None], zg0, zg1]
                    fh = fh * pi[jobs].gather(1, zg0)
                    valid = weights[jobs] >= 1e-6
                    safe_fh = t.where(valid, fh, t.ones_like(fh))
                    terms = safe_fh.log() * weights[jobs]
                    lnL[jobs] = self._sum_last(terms)
                # Original getstats and output parse calls mutate rates again.
                rates2 = self._parse_rates(rates1, start, end)
                _, rs, ra0, ra, mr_stats = self._q_stats(rates2, pi, omega)
                proportion = rs + ra0
                rs0, ra0 = rs / proportion, ra0 / proportion
                sites_s = (rs0 * 3.) * (self.length[start:end] / 3.)
                sites_n = self.length[start:end] - sites_s
                ks = (xb[:, 0] * (rs / mr_stats)) / (3. * rs0)
                ka = (xb[:, 0] * (ra / mr_stats)) / (3. * ra0)
                sd, nd = ks * sites_s, ka * sites_n
                scale = (sd + nd) / self.snp[start:end]
                sd, nd = sd / scale, nd / scale
                rates3 = self._parse_rates(rates2, start, end)
                np = self.nkappa[start:end].to(t.float64) + 2.
                codons = self.length[start:end] / 3.
                aicc = -2. * lnL + ((2. * np) * codons) / (codons - np - 1.)
                time_gy = xb[:, 0] / 3.
                out = output[start:end]
                out[:, :11] = t.stack([lnL, aicc, 3. * time_gy, omega, ka, ks,
                                       sites_s, sites_n, sd, nd, time_gy], dim=1)
                out[:, 11:17] = rates3
                out[:, 17:81] = self.pi[start:end]
                out[:, 81:88] = xb
        self.synchronize()
        return output


def validate(prepared, parameters, outputs, likelihood_backend="eigh"):
    issues, max_abs, max_rel = [], 0., 0.
    for job, result in enumerate(outputs):
        cpu, cpu_status = prepared.pairs[job // 14].evaluate(job % 14, parameters[job])
        for column, (a, b) in enumerate(zip(cpu, result)):
            delta = abs(a - b)
            if math.isfinite(delta):
                max_abs = max(max_abs, delta)
                if a or b:
                    max_rel = max(max_rel, delta / max(abs(a), abs(b)))
            tol = POLICY["absolute_tolerance"] + POLICY["relative_tolerance"] * max(abs(a), abs(b))
            if not math.isfinite(a) or not math.isfinite(b) or delta > tol:
                issues.append({"job": job, "pair": prepared.ids[job // 14],
                               "model": MODELS[job % 14], "column": column,
                               "cpu": a if math.isfinite(a) else str(a),
                               "torch": b if math.isfinite(b) else str(b),
                               "absolute_error": delta if math.isfinite(delta) else str(delta),
                               "tolerance": tol if math.isfinite(tol) else str(tol)})
        if cpu_status[3]:
            issues.append({"job": job, "CPU_eigen_warning_count": cpu_status[3],
                           "torch_backend_diagnostic": "No backend exception; diagnostics are not equivalent to upstream eigen warnings."})
    return {"passed": not issues, "policy": POLICY, "likelihood_backend": likelihood_backend, "issues": issues,
            "issue_count": len(issues), "max_absolute_error": max_abs,
            "max_relative_error": max_rel, "compared_values": len(outputs) * 96}


def run(args):
    if args.output.exists():
        raise ValueError("Output destination already exists")
    args.output.mkdir(parents=True)
    if args.command == "evaluate" and args.device.startswith("cuda"):
        # Inventory BEFORE importing torch or creating a CUDA context.
        smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=True)
        (args.output / "nvidia-smi-before.txt").write_text(smi.stdout, encoding="utf-8")
        csv = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
                              "--format=csv"], text=True, capture_output=True, check=True)
        (args.output / "nvidia-smi-before.csv").write_text(csv.stdout, encoding="utf-8")
    k = kernel32()
    cpu_topology = topology(k)
    process = k.GetCurrentProcess()
    old_mask, _ = affinity(k, process)
    if not k.SetProcessAffinityMask(process, cpu_topology["selected_mask"]):
        raise C.WinError()
    if affinity(k, process)[0] != cpu_topology["selected_mask"]:
        raise RuntimeError("CPU affinity did not match 8 physical/16 logical limit")
    started = time.perf_counter()
    try:
        with PreparedPairs(args.sample, args.count, args.adapter, args.pair) as prepared:
            manifest = prepared.manifest()
            manifest["CPU_affinity_mask"] = hex(cpu_topology["selected_mask"])
            manifest["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            manifest["read_parse_prepare_seconds"] = time.perf_counter() - started
            if args.command == "prepare":
                manifest.update({"torch_imported": False, "GPU_execution": False,
                                 "host_data_only": True})
                dump(args.output / "prepare_manifest.json", manifest)
                print(json.dumps({key: manifest[key] for key in
                                  ("pair_count", "input_sha256", "pattern_count", "max_patterns_per_pair",
                                   "one_change_edges", "read_parse_prepare_seconds", "GPU_execution")}, indent=2))
                return 0
            import_start = time.perf_counter()
            import torch
            torch.set_num_threads(16)
            manifest["torch_import_seconds"] = time.perf_counter() - import_start
            manifest["torch_version"] = torch.__version__
            manifest["torch_built_cuda"] = torch.version.cuda
            upload_start = time.perf_counter()
            evaluator = TorchMatrixEvaluator(prepared, args.device, args.batch_tasks, args.reduction, args.solver,
                                             args.likelihood_backend)
            parameters = default_parameters(len(prepared.ids))
            x = evaluator.tensor(parameters)
            evaluator.synchronize()
            manifest["resident_input_upload_seconds"] = time.perf_counter() - upload_start
            manifest["memory_before"] = evaluator.memory_before
            manifest["device"] = str(evaluator.device)
            manifest["solver_preference"] = args.solver
            manifest["solver_preference_applies_to_likelihood"] = args.likelihood_backend == "eigh"
            manifest["batch_tasks"] = args.batch_tasks
            manifest["reduction"] = args.reduction
            manifest["likelihood_backend"] = args.likelihood_backend
            manifest["prototype_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if args.likelihood_backend == "matrix-exp":
                manifest["numerical_path"] = "FP64 torch.linalg.matrix_exp(S*x0/mr) on pi>1e-6 active compressed blocks; S uses upstream lower-only mirroring. Similarity transform and inactive identity embedding are retained; getstats and three rate parses are unchanged. This is distinct from upstream Householder/QL and independently validated."
            else:
                manifest["numerical_path"] = "Batched symmetric torch.linalg.eigh; distinct from upstream Householder/QL. Tree mode also changes reduction and uses matrix multiplication."
            if evaluator.device.type == "cuda":
                properties = torch.cuda.get_device_properties(evaluator.device)
                manifest["GPU"] = {"ordinal": evaluator.device.index, "name": properties.name,
                                   "total_memory": properties.total_memory}
                torch.cuda.reset_peak_memory_stats(evaluator.device)
            eval_seconds = []
            for _ in range(args.repeats):
                tick = time.perf_counter()
                result = evaluator.evaluate(x)
                eval_seconds.append(time.perf_counter() - tick)
            copy_start = time.perf_counter()
            outputs = result.cpu().tolist()
            flat = array("d", (v for row in outputs for v in row))
            if sys.byteorder != "little":
                flat.byteswap()
            (args.output / "matrix_results.f64").write_bytes(flat.tobytes())
            statuses = array("i", (v for job in range(len(outputs)) for v in (job, -2, 0, 0)))
            (args.output / "matrix_statuses.i32").write_bytes(statuses.tobytes())
            manifest["download_output_seconds"] = time.perf_counter() - copy_start
            manifest["first_evaluate_seconds"] = eval_seconds[0]
            manifest["resident_repeat_evaluate_seconds"] = eval_seconds[1:]
            manifest["all_evaluate_seconds"] = eval_seconds
            manifest["end_to_end_seconds"] = time.perf_counter() - started
            manifest["timing_scope"] = "Read frozen AXT, original native preparation, torch import, resident upload, every requested same-x evaluation, last download/binary write. CPU parity validation excluded. Repeats reuse resident input; these are primitive timings, not MA timings."
            if evaluator.device.type == "cuda":
                manifest["GPU_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(evaluator.device)
                manifest["GPU_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(evaluator.device)
            dump(args.output / "run_manifest.json", manifest)
            report = validate(prepared, parameters, outputs, args.likelihood_backend)
            dump(args.output / "primitive_validation.json", report)
            print(json.dumps({"first_evaluate_seconds": eval_seconds[0],
                              "end_to_end_seconds": manifest["end_to_end_seconds"],
                              "validation_passed": report["passed"],
                              "issue_count": report["issue_count"]}, indent=2))
            return 0 if report["passed"] else 1
    except Exception as error:
        dump(args.output / "failure.json", {"schema": "kakscuda.torch_matrix_failure/1",
                                             "exception": type(error).__name__, "message": str(error),
                                             "device": args.device, "command": args.command,
                                             "likelihood_backend": args.likelihood_backend,
                                             "complete_MA": False})
        raise
    finally:
        k.SetProcessAffinityMask(process, old_mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--sample", type=Path, default=ROOT / "Codex_test/benchmark_1000")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--pair")
    parser.add_argument("--adapter", type=Path, default=ROOT / "build/native-adapter-checked/kaks_adapter.dll")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or one explicit ordinal, e.g. cuda:1")
    parser.add_argument("--batch-tasks", type=int, default=896)
    parser.add_argument("--reduction", choices=("tree", "ordered"), default="tree")
    parser.add_argument("--solver", choices=("default", "cusolver", "magma"), default="default")
    parser.add_argument("--likelihood-backend", choices=("eigh", "matrix-exp"), default="eigh",
                        help="Independent FP64 primitive route; both require strict CPU parity validation")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.count < 1 or args.repeats < 1 or args.batch_tasks < 1:
        parser.error("Counts, repeats, and batch size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
