"""Windows-only ctypes host fibers; this module neither imports nor runs CUDA.

The original bounded BFGS pauses at each uncached lfun request. Batch providers
return objectives = -lnL. Use value_kind='lnL' for Torch output column zero.
All calls and close must occur on the creating Python/OS thread.
"""
from __future__ import annotations
from product_v1_platform import dll_directory
import ctypes as C
import os
from pathlib import Path
import threading
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
I32, U32, U64, D64 = C.c_int32, C.c_uint32, C.c_uint64, C.c_double
PI, PU64, PD = C.POINTER(I32), C.POINTER(U64), C.POINTER(D64)


@dataclass(frozen=True)
class RequestBatch:
    metadata: tuple[tuple[int, ...], ...]
    parameters: tuple[tuple[float, ...], ...]
    tickets: tuple[int, ...]
    completed: int

    @property
    def slots(self):
        return tuple(row[0] for row in self.metadata)

    def __len__(self):
        return len(self.tickets)


class FiberOptimizer:
    def __init__(self, dll_path=None, cache_entries=64, max_pairs=16,
                 stack_reserve=131072, stack_commit=4096):
        if os.name != 'nt':
            raise RuntimeError('Native Windows fibers are required')
        self.owner = threading.get_ident()
        self.path = Path(dll_path or ROOT / 'build/fiber-optimizer-checked/fiber_optimizer.dll').resolve()
        self._dirs = [dll_directory(str(p)) for p in
                      (self.path.parent, ROOT / 'envs/cpu-reference/Library/bin')]
        self.dll = C.CDLL(str(self.path))
        d, error = self.dll, C.create_string_buffer(2048)
        d.kaks_fiber_abi_version.argtypes = []
        d.kaks_fiber_abi_version.restype = U32
        if d.kaks_fiber_abi_version() != 1:
            raise RuntimeError('Unknown fiber optimizer ABI')
        common = [C.c_char_p, U32]
        d.kaks_fiber_create.argtypes = [U32, U32, U32, U32, C.POINTER(C.c_void_p)] + common
        d.kaks_fiber_add_pair.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.POINTER(U32)] + common
        d.kaks_fiber_poll.argtypes = [C.c_void_p, U32, PI, PD, PU64, C.POINTER(U32), C.POINTER(U32)] + common
        d.kaks_fiber_submit.argtypes = [C.c_void_p, U32, PI, PU64, PD, PI] + common
        d.kaks_fiber_mock_evaluate.argtypes = [C.c_void_p, U32, PI, PU64, PD, PI] + common
        d.kaks_fiber_results.argtypes = [C.c_void_p, U32, PD, PI] + common
        d.kaks_fiber_cpu_oracle.argtypes = [C.c_void_p, U32, U32, PD, PI] + common
        d.kaks_fiber_metrics.argtypes = [C.c_void_p, U32, PU64] + common
        d.kaks_fiber_close.argtypes = [C.c_void_p] + common
        for name in ('create', 'add_pair', 'poll', 'submit', 'mock_evaluate', 'results', 'cpu_oracle', 'metrics', 'close'):
            getattr(d, 'kaks_fiber_' + name).restype = C.c_int
        self.context, self.pair_ids, self.closed = C.c_void_p(), [], False
        self._check(d.kaks_fiber_create(cache_entries, stack_reserve, stack_commit, max_pairs,
                                       C.byref(self.context), error, len(error)), error)

    @staticmethod
    def _check(code, error):
        if code:
            raise RuntimeError(error.value.decode('utf8', errors='replace'))

    def _open(self):
        if self.closed:
            raise RuntimeError('Fiber optimizer is closed')
        if threading.get_ident() != self.owner:
            raise RuntimeError('Fiber optimizer requires its creating OS thread')

    @property
    def task_count(self):
        return len(self.pair_ids) * 14

    def add_pair(self, pair_id, sequence_a, sequence_b):
        self._open()
        encode = lambda value: value.encode('ascii') if isinstance(value, str) else bytes(value)
        index, error = U32(), C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_add_pair(self.context, encode(pair_id), encode(sequence_a),
                    encode(sequence_b), C.byref(index), error, len(error)), error)
        self.pair_ids.append(pair_id)
        return index.value

    def poll(self, capacity=None):
        self._open()
        capacity = self.task_count if capacity is None else capacity
        if capacity < 1:
            raise ValueError('Add pairs before polling')
        meta, parameters, tickets = (I32 * (capacity * 6))(), (D64 * (capacity * 7))(), (U64 * capacity)()
        count, done, error = U32(), U32(), C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_poll(self.context, capacity, meta, parameters, tickets,
                    C.byref(count), C.byref(done), error, len(error)), error)
        return RequestBatch(tuple(tuple(meta[i*6:i*6+6]) for i in range(count.value)),
                            tuple(tuple(parameters[i*7:i*7+7]) for i in range(count.value)),
                            tuple(tickets[:count.value]), done.value)

    @staticmethod
    def _request_buffers(batch):
        return (I32 * len(batch))(*batch.slots), (U64 * len(batch))(*batch.tickets)

    def submit(self, batch, values, *, value_kind='negative_log_likelihood', eigen_warnings=None):
        self._open()
        values = values.tolist() if hasattr(values, 'tolist') else list(values)
        if len(values) != len(batch):
            raise ValueError('One value is required per request')
        if value_kind == 'lnL':
            values = [-float(v) for v in values]
        elif value_kind != 'negative_log_likelihood':
            raise ValueError("value_kind must be 'lnL' or 'negative_log_likelihood'")
        slots, tickets = self._request_buffers(batch)
        objectives, error = (D64 * len(batch))(*values), C.create_string_buffer(2048)
        warnings = None
        if eigen_warnings is not None:
            eigen_warnings = list(eigen_warnings)
            if len(eigen_warnings) != len(batch):
                raise ValueError('One warning delta is required per request')
            warnings = (I32 * len(batch))(*eigen_warnings)
        self._check(self.dll.kaks_fiber_submit(self.context, len(batch), slots, tickets,
                    objectives, warnings, error, len(error)), error)

    def mock_evaluate(self, batch):
        self._open()
        slots, tickets = self._request_buffers(batch)
        objectives, warnings, error = (D64 * len(batch))(), (I32 * len(batch))(), C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_mock_evaluate(self.context, len(batch), slots, tickets,
                    objectives, warnings, error, len(error)), error)
        return list(objectives), list(warnings)

    def results(self):
        self._open()
        results, status = (D64 * (self.task_count * 96))(), (I32 * (self.task_count * 4))()
        error = C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_results(self.context, self.task_count, results,
                    status, error, len(error)), error)
        return results, status

    def cpu_oracle(self, pair_index, model_index):
        self._open()
        result, status, error = (D64 * 96)(), (I32 * 4)(), C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_cpu_oracle(self.context, pair_index, model_index,
                    result, status, error, len(error)), error)
        return result, status

    def metrics(self):
        self._open()
        values, error = (U64 * (self.task_count * 6))(), C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_metrics(self.context, self.task_count, values, error, len(error)), error)
        return [list(values[i*6:i*6+6]) for i in range(self.task_count)]

    def close(self):
        if not self.closed:
            self._open()
            error = C.create_string_buffer(2048)
            self._check(self.dll.kaks_fiber_close(self.context, error, len(error)), error)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
