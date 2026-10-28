"""Graph-safe FP64 GY94 likelihood prototype, independent of legacy QL parity.

T18 coefficients and theta are from PyTorch v2.5.1 LinearAlgebra.cpp. Unlike
torch.linalg.matrix_exp, scaling uses no device-to-host decisions. Every graph
executes a fixed number of squarings. Out-of-range rows are marked invalid and
either rejected or evaluated by torch.linalg.matrix_exp outside the graph.
Original host optimizer iteration/convergence rules are not changed here.
"""
from __future__ import annotations

import math
import time
from torch_matrix_prototype import TorchMatrixEvaluator, NKAPPA, RATE_MAP

SOURCE_URL = "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/LinearAlgebra.cpp"
THETA18_FP64 = 1.090863719290036
T18_COEFFICIENTS = (
    (0., -1.00365581030144618291e-1, -8.02924648241156932449e-3, -8.92138498045729985177e-4, 0.),
    (0., 3.97849749499645077844e-1, 1.36783778460411720168, 4.98289622525382669416e-1, -6.37898194594723280150e-4),
    (-1.09676396052962061844e1, 1.68015813878906206114, 5.71779846478865511061e-2, -6.98210122488052056106e-3, 3.34975017086070470649e-5),
    (-9.04316832390810593223e-2, -6.76404519071381882256e-2, 6.75961301770459654925e-2, 2.95552570429315521194e-2, -1.39180257516060693404e-5),
    (0., 0., -9.23364619367118555360e-2, -1.69364939002081722752e-2, -1.40086798182036094347e-5),
)


def fixed_t18(a, coefficients, identity, max_squarings=6):
    """Return (exp, valid, norm1), with graph-static launches and FP64 math.

    Invalid input is replaced by zero for the internal polynomial only; its
    returned exponential is NaN. The caller must reject it or explicitly use
    another validated exponential. No truncated exponential is accepted.
    """
    import torch as t
    norm = a.abs().sum(dim=-2).amax(dim=-1)
    s = t.ceil(t.log2(norm / THETA18_FP64)).clamp_min(0.)
    valid = t.isfinite(norm) & (s <= max_squarings)
    safe_s = t.where(valid, s, t.zeros_like(s))
    safe_a = t.where(valid[:, None, None], a, t.zeros_like(a))
    scaled = safe_a * t.pow(2., -safe_s)[:, None, None]
    # The five products and linear-combination order match the T18 polynomial.
    a2 = scaled @ scaled
    a3 = scaled @ a2
    a6 = a3 @ a3
    powers = t.stack((identity.expand_as(a), scaled, a2, a3, a6))
    bs = t.ops.aten._compute_linear_combination(powers, coefficients)
    b3 = bs[3] + bs[0] @ bs[4]
    result = bs[1] + (bs[2] + b3) @ b3
    for i in range(max_squarings):
        square = result @ result
        result = t.where((safe_s > i)[:, None, None], square, result)
    result = t.where(valid[:, None, None], result, t.full_like(result, float("nan")))
    return result, valid, norm


class TorchGraphBatchLikelihood(TorchMatrixEvaluator):
    """Indexed objective provider with lazy CUDA graph buckets.

    Input codons/pi remain resident. Request buffers have fixed addresses per
    bucket; only slot IDs and seven doubles per request are uploaded. Duplicate
    padding is pure evaluation and never reaches optimizer result submission.
    Graphs and request calls are single-threaded and cannot overlap.
    """
    def __init__(self, prepared, device="cuda:1", batch_tasks=256,
                 max_squarings=6, use_graphs=True, fallback="torch", adaptive_squarings=True,
                 bucket_quantum=32):
        if max_squarings < 0 or max_squarings > 16:
            raise ValueError("max_squarings must be in [0,16]")
        if fallback not in ("raise", "torch"):
            raise ValueError("fallback must be raise or torch")
        if bucket_quantum < 1:
            raise ValueError("bucket_quantum must be positive")
        # All allocated buckets can occupy almost twice the largest bucket.
        largest = 1 << (max(1, batch_tasks) - 1).bit_length()
        super().__init__(prepared, device, largest * 2, "tree", "default", "matrix-exp")
        self.batch_tasks = batch_tasks
        self.max_squarings = max_squarings
        self.adaptive_squarings = adaptive_squarings
        self.bucket_quantum = bucket_quantum
        self.use_graphs = use_graphs and self.device.type == "cuda"
        self.fallback = fallback
        self.identity = self.tensor([[float(i == j) for j in range(61)] for i in range(61)])
        self.coefficients = self.tensor(T18_COEFFICIENTS)
        self.buckets = {}
        self.requests = self.parameter_transfer_bytes = self.objective_transfer_bytes = 0
        self.fallback_requests = self.fallback_calls = 0
        self.graph_capture_seconds = self.graph_captures = self.graph_replays = 0
        self.maximum_observed_norm = 0.
        self.maximum_bucket = largest
        self.unified = True
        self._prepare_host_bounds()

    def _prepare_host_bounds(self):
        """Twelve positive basis norm bounds and mr coefficients per pi type.

        Each basis includes the full Q diagonal followed by the active mask.
        max(column sums) of each basis, then triangle inequality, gives an
        upper bound; evaluating it needs only 12 terms per requested model.
        """
        def coefficients(pi):
            live = [p > 1e-6 for p in pi]
            columns = [[0.] * 12 for _ in range(61)]
            mr = [0.] * 12
            for i, j, rate, syn in self.prepared.edges:
                k = rate + (0 if syn else 6)
                symmetric = math.sqrt(pi[i] * pi[j]) if live[i] and live[j] else 0.
                if live[i]:
                    columns[i][k] += pi[j] + symmetric
                if live[j]:
                    columns[j][k] += pi[i] + symmetric
                mr[k] += (2. * pi[i]) * pi[j]
            return [max(row[k] for row in columns) for k in range(12)], mr
        equal = coefficients([1. / 61] * 61)
        self.host_bounds = []
        for values in self.prepared.values:
            unequal = coefficients(values[2:63])
            self.host_bounds.append((equal, unequal))

    def _choose_squarings(self, slots, parameters):
        if not self.adaptive_squarings:
            return self.max_squarings
        maximum = 0.
        for slot, x in zip(slots, parameters):
            model = slot % 14
            nk = NKAPPA[model]
            raw = [x[i + 1] if i < nk else 1. for i in range(6)]
            rates = [raw[i] for i in RATE_MAP[model // 2]]
            omega = x[nk + 1]
            if x[0] < 0. or omega <= 0. or min(rates) <= 0.:
                return self.max_squarings
            norm_coeff, mr_coeff = self.host_bounds[slot // 14][model % 2]
            weights = rates + [v * omega for v in rates]
            mr = sum(c * w for c, w in zip(mr_coeff, weights))
            if mr <= 0. or not math.isfinite(mr):
                return self.max_squarings
            bound = (x[0] / mr) * sum(c * w for c, w in zip(norm_coeff, weights))
            maximum = max(maximum, bound)
        # The tensor guard is authoritative. Roundoff near a threshold can
        # only cause explicit fallback, never an accepted truncated result.
        if not math.isfinite(maximum):
            return self.max_squarings
        if maximum <= THETA18_FP64 / (1. + 1e-12):
            return 0
        return min(self.max_squarings, max(0, math.ceil(math.log2(
            maximum * (1. + 1e-12) / THETA18_FP64))))

    def _matrix_inputs(self, indices, x):
        t = self.torch
        pi = self.pi[indices, :61]
        omega = x.gather(1, (self.nkappa[indices] + 1)[:, None])[:, 0]
        raw = t.where(self.active_parameter[indices], x[:, 1:7], t.ones_like(x[:, 1:7]))
        rates = raw.gather(1, self.rate_map[indices])
        # mr is calculated from the full Q before inactive rows are zeroed.
        q, _, _, _, mr = self._q_stats(rates, pi, omega)
        live = pi > 1e-6
        sqrt_pi = t.where(live, pi, t.ones_like(pi)).sqrt()
        s = (q * sqrt_pi[:, :, None]) / sqrt_pi[:, None, :]
        lower = t.tril(s, diagonal=-1)
        symmetric = lower + lower.transpose(-1, -2) + t.diag_embed(q.diagonal(dim1=-2, dim2=-1))
        live_outer = live[:, :, None] & live[:, None, :]
        symmetric = t.where(live_outer, symmetric, t.zeros_like(symmetric))
        a = (symmetric * x[:, 0, None, None]) / mr[:, None, None]
        return a, pi, sqrt_pi, live_outer

    def _objective_from_exp(self, indices, exponential, pi, sqrt_pi, live_outer):
        t = self.torch
        transition = (exponential / sqrt_pi[:, :, None]) * sqrt_pi[:, None, :]
        transition = t.where(live_outer, transition, self.identity)
        pair = indices // 14
        z0, z1, weights = self.z0[pair], self.z1[pair], self.weight[pair]
        rows = t.arange(indices.shape[0], device=self.device)[:, None]
        fh = transition[rows, z0, z1] * pi.gather(1, z0)
        safe = t.where(weights >= 1e-6, fh, t.ones_like(fh))
        return -(safe.log() * weights).sum(dim=-1)

    def _forward(self, indices, x, squarings):
        t = self.torch
        a, pi, sqrt_pi, live_outer = self._matrix_inputs(indices, x)
        exponential, valid, norm = fixed_t18(a, self.coefficients, self.identity, squarings)
        objective = self._objective_from_exp(indices, exponential, pi, sqrt_pi, live_outer)
        # One result transfer carries both values and guard diagnostics.
        return t.stack((objective, valid.to(t.float64), norm), dim=-1)

    def _new_bucket(self, size, squarings, initial_indices, initial_x):
        t = self.torch
        started = time.perf_counter()
        indices = initial_indices.clone()
        x = initial_x.clone()
        if not self.use_graphs:
            bucket = {"indices": indices, "x": x, "graph": None}
        else:
            free, _ = t.cuda.mem_get_info(self.device)
            if size * 61 * 61 * 8 * 48 + 2 * 1024 ** 3 > free:
                raise MemoryError("New graph workspace plus 2 GiB reserve exceeds current free GPU memory")
            stream = t.cuda.Stream(device=self.device)
            stream.wait_stream(t.cuda.current_stream(self.device))
            with t.cuda.stream(stream):
                for _ in range(3):
                    self._forward(indices, x, squarings)
            t.cuda.current_stream(self.device).wait_stream(stream)
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph, stream=stream):
                result = self._forward(indices, x, squarings)
            bucket = {"indices": indices, "x": x, "graph": graph, "result": result,
                      "stream": stream}
            self.graph_captures += 1
        self.buckets[(size, squarings)] = bucket
        self.graph_capture_seconds += time.perf_counter() - started
        return bucket

    def _fallback_values(self, slots, parameters):
        t = self.torch
        indices, x = self.tensor(slots, t.int64), self.tensor(parameters)
        a, pi, sqrt_pi, live_outer = self._matrix_inputs(indices, x)
        exp = t.linalg.matrix_exp(a)
        result = self._objective_from_exp(indices, exp, pi, sqrt_pi, live_outer)
        self.fallback_requests += len(slots)
        self.fallback_calls += 1
        self.parameter_transfer_bytes += len(slots) * (7 * 8 + 8)
        self.objective_transfer_bytes += len(slots) * 8
        return result.cpu().tolist()

    def objectives(self, slots, parameters):
        """Return positive negative log likelihood in the original slot order."""
        if len(slots) != len(parameters):
            raise ValueError("Slot and parameter counts differ")
        if len(set(slots)) != len(slots) or any(s < 0 or s >= self.task_count for s in slots):
            raise ValueError("Request slots must be unique and valid")
        if any(len(x) != 7 or any(not math.isfinite(v) for v in x) for x in parameters):
            raise ValueError("Parameters must contain seven finite doubles per slot")
        t = self.torch
        values = []
        with t.inference_mode():
            for offset in range(0, len(slots), self.batch_tasks):
                local_slots = list(slots[offset:offset + self.batch_tasks])
                local_x = list(parameters[offset:offset + self.batch_tasks])
                count = len(local_slots)
                size = (1 << (count - 1).bit_length()) if count <= self.bucket_quantum else (
                    (count + self.bucket_quantum - 1) // self.bucket_quantum) * self.bucket_quantum
                squarings = self._choose_squarings(local_slots, local_x)
                padded_slots = local_slots + [local_slots[0]] * (size - count)
                padded_x = local_x + [local_x[0]] * (size - count)
                incoming_indices = self.tensor(padded_slots, t.int64)
                incoming_x = self.tensor(padded_x)
                bucket = self.buckets.get((size, squarings))
                if bucket is None:
                    bucket = self._new_bucket(size, squarings, incoming_indices, incoming_x)
                else:
                    bucket["indices"].copy_(incoming_indices)
                    bucket["x"].copy_(incoming_x)
                if bucket["graph"] is None:
                    result = self._forward(bucket["indices"], bucket["x"], squarings)
                else:
                    bucket["graph"].replay()
                    self.graph_replays += 1
                    result = bucket["result"]
                host = result[:count].cpu().tolist()
                bad = [i for i, row in enumerate(host) if row[1] != 1.]
                for row in host:
                    if math.isfinite(row[2]):
                        self.maximum_observed_norm = max(self.maximum_observed_norm, row[2])
                if bad:
                    if self.fallback == "raise":
                        raise ArithmeticError("T18 scaling guard exceeded: " + str(
                            [(local_slots[i], host[i][2], squarings) for i in bad]))
                    fallback = self._fallback_values([local_slots[i] for i in bad], [local_x[i] for i in bad])
                    for i, f in zip(bad, fallback):
                        host[i][0] = f
                if any(not math.isfinite(row[0]) for row in host):
                    raise ArithmeticError("Nonfinite T18 or fallback likelihood")
                values.extend(row[0] for row in host)
                self.parameter_transfer_bytes += size * (7 * 8 + 8)
                self.objective_transfer_bytes += count * 3 * 8
        self.requests += len(slots)
        return values

    def graph_manifest(self):
        return {"schema": "kakscuda.torch_graph_likelihood/1", "source": SOURCE_URL,
                "dtype": "float64", "theta18": THETA18_FP64, "max_squarings": self.max_squarings,
                "scaling_guard_limit": THETA18_FP64 * 2 ** self.max_squarings,
                "fallback": self.fallback, "fallback_requests": self.fallback_requests,
                "fallback_calls": self.fallback_calls, "maximum_observed_norm": self.maximum_observed_norm,
                "use_graphs": self.use_graphs, "adaptive_squarings": self.adaptive_squarings,
                "bucket_quantum": self.bucket_quantum,
                "bucket_sizes_and_squarings": sorted(self.buckets),
                "graph_captures": self.graph_captures, "graph_replays": self.graph_replays,
                "graph_capture_seconds": self.graph_capture_seconds,
                "requests": self.requests, "parameter_transfer_bytes": self.parameter_transfer_bytes,
                "objective_transfer_bytes": self.objective_transfer_bytes,
                "optimizer_changed": False, "legacy_QL_bit_parity_claimed": False}
