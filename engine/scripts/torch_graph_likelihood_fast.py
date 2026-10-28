"""Independent v2: native host bounds, reused pinned requests, Q/mr only.

T18, the graph scaling guard, masks, reduction order, and optimizer contract
are inherited from the validated v1. No original source or v1 file is edited.
"""
from __future__ import annotations
from product_v1_platform import native_path
import ctypes as C
import hashlib
import math
from pathlib import Path
import time

from torch_graph_likelihood import TorchGraphBatchLikelihood

ROOT = Path(__file__).resolve().parents[1]


class NativeGraphHost:
    def __init__(self, torch, host_bounds, dll_path=None):
        self.torch = torch
        self.path = native_path(Path(dll_path or ROOT / "build/torch-graph-host/torch_graph_host.dll"))
        self.dll = C.CDLL(str(self.path))
        d = self.dll
        self.error = C.create_string_buffer(512)
        self.context = C.c_void_p()
        self.squarings = C.c_int()
        self.maximum = C.c_double()
        self.bad_count = C.c_uint32()
        self.bad_rows = None
        self.bad_capacity = 0
        common = [C.c_char_p, C.c_uint32]
        d.gy94_graph_host_create.argtypes = [C.c_uint32, C.c_void_p, C.POINTER(C.c_void_p)] + common
        d.gy94_graph_host_free.argtypes = [C.c_void_p]
        d.gy94_graph_host_free.restype = None
        d.gy94_graph_host_validate.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32] + common
        d.gy94_graph_host_choose.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32,
                                            C.c_int, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_double)] + common
        d.gy94_graph_host_analyze.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p,
                                             C.POINTER(C.c_uint32), C.POINTER(C.c_double)] + common
        for name in ("create", "validate", "choose", "analyze"):
            getattr(d, "gy94_graph_host_" + name).restype = C.c_int
        flat = [[[v for column in item for v in column] for item in pair] for pair in host_bounds]
        bounds = torch.tensor(flat, dtype=torch.float64)
        self._check(d.gy94_graph_host_create(len(host_bounds), bounds.data_ptr(), C.byref(self.context),
                                             self.error, len(self.error)))
        self.dll_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def _check(self, code):
        if code:
            raise ValueError(self.error.value.decode("utf-8", errors="replace"))

    def validate(self, indices, x):
        self._check(self.dll.gy94_graph_host_validate(self.context, indices.data_ptr(), x.data_ptr(),
                                                      len(indices), self.error, len(self.error)))

    def choose(self, indices, x, limit, adaptive):
        self._check(self.dll.gy94_graph_host_choose(self.context, indices.data_ptr(), x.data_ptr(),
            len(indices), limit, adaptive, C.byref(self.squarings), C.byref(self.maximum),
            self.error, len(self.error)))
        return self.squarings.value

    def analyze(self, result):
        count = len(result)
        if count > self.bad_capacity:
            self.bad_rows = (C.c_int32 * count)()
            self.bad_capacity = count
        self._check(self.dll.gy94_graph_host_analyze(result.data_ptr(), count, self.bad_rows,
            C.byref(self.bad_count), C.byref(self.maximum), self.error, len(self.error)))
        return list(self.bad_rows[:self.bad_count.value]), self.maximum.value

    def close(self):
        if self.context.value:
            self.dll.gy94_graph_host_free(self.context)
            self.context.value = None

    def __del__(self):
        if hasattr(self, "context"):
            self.close()


class TorchGraphBatchLikelihoodFast(TorchGraphBatchLikelihood):
    def __init__(self, prepared, device="cuda:1", batch_tasks=256,
                 max_squarings=6, use_graphs=True, fallback="torch", adaptive_squarings=True,
                 bucket_quantum=32, host_dll=None):
        if batch_tasks < 1:
            raise ValueError("batch_tasks must be positive")
        super().__init__(prepared, device, batch_tasks, max_squarings, use_graphs,
                         fallback, adaptive_squarings, bucket_quantum)
        self.native_host = NativeGraphHost(self.torch, self.host_bounds, host_dll)
        self.native_bound_seconds = self.host_marshal_validate_seconds = 0.

    def _q_mr(self, rates, pi, omega):
        t = self.torch
        lower = rates[:, self.edge_rate] * pi[:, self.edge_j]
        upper = rates[:, self.edge_rate] * pi[:, self.edge_i]
        pij = (2. * pi[:, self.edge_i]) * lower
        zeros = t.zeros_like(pij)
        # Deliberately retain the two reductions then rs+ra, exactly as v1.
        rs = self._sum_last(t.where(self.edge_syn[None, :], pij, zeros))
        ra = self._sum_last(t.where(self.edge_syn[None, :], zeros, pij * omega[:, None]))
        mr = rs + ra
        multiplier = t.where(self.edge_syn[None, :], t.ones_like(pij), omega[:, None])
        q = t.zeros((len(pi), 61, 61), dtype=t.float64, device=self.device)
        q[:, self.edge_i, self.edge_j] = lower * multiplier
        q[:, self.edge_j, self.edge_i] = upper * multiplier
        q.diagonal(dim1=-2, dim2=-1).copy_(-self._sum_last(q))
        return q, mr

    def _matrix_inputs(self, indices, x):
        t = self.torch
        pi = self.pi[indices, :61]
        omega = x.gather(1, (self.nkappa[indices] + 1)[:, None])[:, 0]
        raw = t.where(self.active_parameter[indices], x[:, 1:7], t.ones_like(x[:, 1:7]))
        rates = raw.gather(1, self.rate_map[indices])
        q, mr = self._q_mr(rates, pi, omega)
        live = pi > 1e-6
        sqrt_pi = t.where(live, pi, t.ones_like(pi)).sqrt()
        s = (q * sqrt_pi[:, :, None]) / sqrt_pi[:, None, :]
        lower = t.tril(s, diagonal=-1)
        symmetric = lower + lower.transpose(-1, -2) + t.diag_embed(q.diagonal(dim1=-2, dim2=-1))
        live_outer = live[:, :, None] & live[:, None, :]
        symmetric = t.where(live_outer, symmetric, t.zeros_like(symmetric))
        a = (symmetric * x[:, 0, None, None]) / mr[:, None, None]
        return a, pi, sqrt_pi, live_outer

    def _new_bucket(self, size, squarings, initial_indices, initial_x):
        # A fixed margin includes small-batch stream/BLAS workspace overhead.
        t = self.torch
        if self.use_graphs:
            free, _ = t.cuda.mem_get_info(self.device)
            if size * 61 * 61 * 8 * 48 + 64 * 1024 ** 2 + 2 * 1024 ** 3 > free:
                raise MemoryError("Graph workspace, fixed margin and 2 GiB reserve exceed free GPU memory")
        bucket = super()._new_bucket(size, squarings, initial_indices, initial_x)
        pinned = self.device.type == "cuda"
        bucket["host_indices"] = t.empty(size, dtype=t.int64, pin_memory=pinned)
        bucket["host_x"] = t.empty((size, 7), dtype=t.float64, pin_memory=pinned)
        return bucket

    def objectives(self, slots, parameters):
        if len(slots) != len(parameters):
            raise ValueError("Slot and parameter counts differ")
        if not len(slots):
            return []
        t = self.torch
        started = time.perf_counter()
        host_indices = t.tensor(slots, dtype=t.int64, device="cpu")
        host_x = t.tensor(parameters, dtype=t.float64, device="cpu")
        if host_indices.shape != (len(slots),) or host_x.shape != (len(slots), 7):
            raise ValueError("Parameters must contain seven doubles per slot")
        self.native_host.validate(host_indices, host_x)
        self.host_marshal_validate_seconds += time.perf_counter() - started
        values = []
        with t.inference_mode():
            for offset in range(0, len(slots), self.batch_tasks):
                indices = host_indices[offset:offset + self.batch_tasks]
                x = host_x[offset:offset + self.batch_tasks]
                count = len(indices)
                size = (1 << (count - 1).bit_length()) if count <= self.bucket_quantum else (
                    (count + self.bucket_quantum - 1) // self.bucket_quantum) * self.bucket_quantum
                started = time.perf_counter()
                squarings = self.native_host.choose(indices, x, self.max_squarings, self.adaptive_squarings)
                self.native_bound_seconds += time.perf_counter() - started
                bucket = self.buckets.get((size, squarings))
                if bucket is None:
                    initial_indices = t.empty(size, dtype=t.int64, device=self.device)
                    initial_x = t.empty((size, 7), dtype=t.float64, device=self.device)
                    initial_indices[:count].copy_(indices); initial_x[:count].copy_(x)
                    if count < size:
                        initial_indices[count:].copy_(initial_indices[:1].expand(size-count))
                        initial_x[count:].copy_(initial_x[:1].expand(size-count,7))
                    bucket = self._new_bucket(size, squarings, initial_indices, initial_x)
                    # Initial upload was real rows only; device padding stays
                    # on device. Later pinned uploads include padding.
                    transferred = count * 64
                else:
                    bucket["host_indices"][:count].copy_(indices)
                    bucket["host_x"][:count].copy_(x)
                    if count < size:
                        bucket["host_indices"][count:].copy_(indices[:1].expand(size-count))
                        bucket["host_x"][count:].copy_(x[:1].expand(size-count,7))
                    bucket["indices"].copy_(bucket["host_indices"], non_blocking=True)
                    bucket["x"].copy_(bucket["host_x"], non_blocking=True)
                    transferred = size * 64
                if bucket["graph"] is None:
                    result = self._forward(bucket["indices"], bucket["x"], squarings)
                else:
                    bucket["graph"].replay()
                    self.graph_replays += 1
                    result = bucket["result"]
                host_result = result[:count].cpu().contiguous()
                bad, maximum = self.native_host.analyze(host_result)
                self.maximum_observed_norm = max(self.maximum_observed_norm, maximum)
                if bad:
                    if self.fallback == "raise":
                        raise ArithmeticError("T18 scaling guard exceeded at " + str(
                            [(slots[offset+i], host_result[i,2].item(), squarings) for i in bad]))
                    fallback = self._fallback_values([slots[offset+i] for i in bad],
                                                     [parameters[offset+i] for i in bad])
                    for i, value in zip(bad, fallback):
                        if not math.isfinite(value):
                            raise ArithmeticError("Nonfinite fallback likelihood")
                        host_result[i,0] = value
                values.extend(host_result[:,0].tolist())
                self.parameter_transfer_bytes += transferred
                self.objective_transfer_bytes += count * 24
        self.requests += len(slots)
        return values

    def graph_manifest(self):
        report = super().graph_manifest()
        report.update({"schema": "kakscuda.torch_graph_likelihood_fast/2",
                       "native_host_DLL": str(self.native_host.path),
                       "native_host_DLL_sha256": self.native_host.dll_sha256,
                       "native_bound_seconds": self.native_bound_seconds,
                       "host_marshal_validate_seconds": self.host_marshal_validate_seconds,
                       "Q_unused_ra0_reduction_removed": True,
                       "fixed_graph_workspace_margin_bytes": 64 * 1024**2,
                       "request_buffers": "CPU tensors marshal once per wave; pinned fixed staging per graph"})
        return report
