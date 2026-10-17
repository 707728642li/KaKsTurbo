"""Development-only CUDA Driver transport around the unchanged round2 FP64 tape.

The allocation/stream API is explicitly NativeCUDAArrayAPI, never Torch or a
replacement Python module. NativeOptimizedT18Runtime.forward_gradient and every
CUDA/PTX equation are inherited. Only the sparse constructor omits creation of
unused Torch AD decorators. Calls return every original request in its original
ordinal, including repeated slots; the optimizer remains outside this module.
"""
from __future__ import annotations
from product_v1_platform import build_record, cuda_driver, native_path, snapshot_directory

from array import array
from collections import OrderedDict
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from cuda_array_round3 import NativeCUDAArrayAPI
from gpu_coalesced_gemm_round2 import CoalescedGemmRuntime
from gpu_matrix_structural_zero_round2 import StructuralZeroForwardMatrixRuntime
from gpu_native_intscale_t18_round2 import NativeIntScaleT18Helpers
from gpu_native_objective_round2 import NativeObjectiveRuntime
from gpu_sparse_t18_round2 import SparseRuntime, structural_csr, build_sparse_ptx
from gpu_t18_fusion_round2 import FusionRuntime
from publication_gpu_gradient import GPUGradientError
from publication_gpu_gradient_v2 import GraphWorkspaceBudgetError
from publication_gpu_likelihood import PublicationGPULikelihood
from publication_gpu_native_optimized_t18_round2 import NativeOptimizedT18Runtime
from publication_gpu_tight_round2 import TightContext
from torch_graph_likelihood_fast import NativeGraphHost
from torch_matrix_prototype import NKAPPA, RATE_MAP, codon_edges

ROOT = Path(__file__).resolve().parents[1]
HOST_DLL = native_path(ROOT / 'build/publication-graph-host-v1/torch_graph_host.dll')
STAGING_DLL = native_path(ROOT / 'build/driver-staging-round3-v1/driver_staging.dll')
RESERVE = 2 * 1024**3
MAX_REQUESTS = 1000000
# All are existing, independently gated round2 builds. Refuse a missing build
# before any constructor can invoke its legacy CPU NVRTC build-on-demand path.
REQUIRED_BUILDS = (
    ('gpu-matrix-contract-round2-v1', 'matrix_contract.ptx'),
    ('gpu-matrix-structural-zero-round2-fp64-v1', 'matrix_structural.ptx'),
    ('gpu-gemm-coalesced-round2-fp64-v1', 'gemm_coalesced.ptx'),
    ('gpu-sparse-t18-round2-fp64-v1', 'sparse_t18.ptx'),
    ('gpu-t18-fusion-round2-v1', 't18_fold.ptx'),
    ('gpu-native-intscale-t18-round2-fp64-v1', 'native_t18_intscale.ptx'),
    ('gpu-native-objective-round2-fp64-v1', 'native_objective.ptx'),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _torch_absent():
    return not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)


class _CPUHostArray:
    """NativeGraphHost's constructor needs an ordinary host FP64 pointer."""
    def __init__(self, values):
        flat = []
        def append(value):
            if isinstance(value, (list, tuple)):
                for item in value:
                    append(item)
            else:
                flat.append(value)
        append(values)
        self.values = (C.c_double * len(flat))(*flat)

    def data_ptr(self):
        return C.addressof(self.values)


class _CPUHostFactory:
    float64 = 'native-host-float64'

    @staticmethod
    def tensor(values, dtype):
        if dtype != _CPUHostFactory.float64:
            raise ValueError('Native host bounds require host FP64')
        return _CPUHostArray(values)


class SparseDriverRuntime(SparseRuntime):
    """Same module/product methods; no unused Torch AD decorator constructor."""
    def __init__(self, api, device, dense_runtime, edges=None):
        self.torch, self.device, self.dense = api, api.device(device), dense_runtime
        if self.device.type != 'cuda' or self.device.index is None:
            raise ValueError('SparseDriverRuntime requires an explicit CUDA device')
        structural_csr(edges)
        ptx, self.metadata = build_sparse_ptx()
        self.path = ROOT / 'build/gpu-sparse-t18-round2-fp64-v1'
        self.artifacts = {name: sha(self.path / name) for name in
            ('build.json', 'sparse_t18.cu', 'sparse_t18_template.cu', 'structural_csr.json', 'sparse_t18.ptx')}
        self.driver = cuda_driver()
        self.module = C.c_void_p()
        self._bind('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p])
        self._bind('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
        self._bind('cuLaunchKernel', [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.c_void_p, C.c_void_p])
        self._bind('cuModuleUnload', [C.c_void_p])
        with api.cuda.device(self.device):
            api.cuda.current_stream(self.device)
            blob = C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module), blob))
            self.functions = {}
            try:
                for name in ('sparse_left', 'sparse_right'):
                    function = C.c_void_p()
                    self._check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()))
                    self.functions[name] = function
            except BaseException as error:
                cleanup = self.driver.cuModuleUnload(self.module)
                if cleanup:
                    raise RuntimeError('Sparse lookup and partial unload failed: ' + str(cleanup)) from error
                self.module = C.c_void_p()
                raise
        # Deliberately omit make_sparse_products: no AD apply is used by the
        # inherited full-native tape, and its absolute decorator import is Torch.


class NativeDriverT18Runtime(NativeOptimizedT18Runtime):
    """Constructor selection only; full FP64 forward/reverse method inherited."""
    def __init__(self, owner, objective_runtime):
        self.owner, self.torch, self.device = owner, owner.torch, owner.device
        self.matrix = StructuralZeroForwardMatrixRuntime(owner)
        self.gemm = CoalescedGemmRuntime(self.torch, self.device, 32)
        self.sparse = SparseDriverRuntime(self.torch, self.device, self.gemm, owner.prepared.edges)
        self.coefficients = FusionRuntime(self.torch, self.device)
        self.helpers = NativeIntScaleT18Helpers(self.torch, self.device)
        self.objective = objective_runtime

    def manifest(self):
        report = super().manifest()
        report['Torch_scope'] = 'None; explicit NativeCUDAArrayAPI allocations/views/Driver stream and graphs'
        report['array_adapter'] = 'NativeCUDAArrayAPI; no Torch import or fake module'
        report['sparse_constructor_only_change'] = 'Unused absolute Torch AD decorator creation omitted'
        return report


class _Staging:
    """Checked, byte-only stable split/pack/scatter from root's new native DLL."""
    def __init__(self):
        self.path = STAGING_DLL
        self.metadata = json.loads((build_record(self.path)).read_text(encoding='utf-8'))
        self.files = [self.path, build_record(self.path)]
        if sha(self.path) != self.metadata['dll_sha256']:
            raise ValueError('Driver staging DLL differs from its build record')
        for name, expected in self.metadata['source_sha256'].items():
            live = ROOT / 'src/gpu_round3' / name
            snapshot = snapshot_directory(self.path) / name
            if sha(live) != expected or sha(snapshot) != expected:
                raise ValueError('Driver staging source/snapshot mismatch')
            self.files.extend((live, snapshot))
        self.before = {str(path.relative_to(ROOT)): sha(path) for path in self.files}
        self.dll = C.CDLL(str(self.path))
        self.dll.drv_split.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        self.dll.drv_pack.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32,
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p, C.c_void_p]
        self.dll.drv_scatter.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32,
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        for name in ('drv_split', 'drv_pack', 'drv_scatter'):
            getattr(self.dll, name).restype = C.c_int

    @staticmethod
    def check(code, operation):
        if code:
            raise ValueError('Driver staging ' + operation + ' rejected metadata, code ' + str(code))

    def manifest(self):
        after = {str(path.relative_to(ROOT)): sha(path) for path in self.files}
        if after != self.before:
            raise RuntimeError('Driver staging source/snapshot/build/DLL changed')
        return dict(self.metadata, source_snapshot_build_DLL_before=self.before,
            source_snapshot_build_DLL_after=after, artifact_unchanged=True)


class PublicationGPUGradient:
    """Single-owner synchronous wave API with resident CUDA Driver graph pools.

    Kind0 returns raw F and zero7; kind1 returns raw F and x*dF/draw_x, never
    divided by codon L. Borrowed ctypes outputs live until the next public call.
    All subgroups finish before original-order scatter and optimizer submission.
    Numeric failed rows are explicit NaN/code1 with successful rows preserved.
    """
    def __init__(self, prepared, device='cuda:1', batch_tasks=512,
                 max_cached_buckets=128, graph_min_rows=32, graph_warmups=1,
                 max_squarings=6, adaptive_squarings=True, use_graphs=True,
                 profile_execution=False, device_uuid=None, expected_uuid=None,
                 fallback='raise'):
        if not _torch_absent():
            raise RuntimeError('CUDA Driver provider requires a process without Torch')
        if fallback != 'raise':
            raise ValueError('Driver provider uses explicit numerical row failure only')
        for value, label, maximum in ((batch_tasks, 'batch_tasks', 1024),
                                     (graph_min_rows, 'graph_min_rows', 1024)):
            if type(value) is not int or not 1 <= value <= maximum or value & (value - 1):
                raise ValueError(label + ' must be a positive power of two <=1024')
        if graph_min_rows > batch_tasks:
            raise ValueError('graph_min_rows must not exceed batch_tasks')
        if type(max_cached_buckets) is not int or max_cached_buckets < 1:
            raise ValueError('Positive graph cache capacity required')
        if type(graph_warmups) is not int or not 1 <= graph_warmups <= 3:
            raise ValueError('graph_warmups must be1..3')
        if type(max_squarings) is not int or not 0 <= max_squarings <= 16:
            raise ValueError('max_squarings must be0..16')
        if len(prepared.ids) != len(prepared.values) or not 1 <= len(prepared.values) <= 1000:
            raise ValueError('This isolated development provider requires1..1000 complete pairs')
        if [tuple(edge) for edge in prepared.edges] != codon_edges():
            raise ValueError('Original code1 edge/order contract required')
        if device_uuid and expected_uuid and self._canonical_uuid(device_uuid) != self._canonical_uuid(expected_uuid):
            raise ValueError('Two different expected device UUIDs')
        for directory, ptx in REQUIRED_BUILDS:
            for name in ('build.json', ptx):
                if not (ROOT / 'build' / directory / name).is_file():
                    raise RuntimeError('Existing gated PTX build required; Driver provider cannot compile: ' + directory)
        self.owner_thread = threading.get_ident()
        self._public_lock = threading.Lock()
        self._closed = self._quarantined = False
        self.prepared = prepared
        self.batch_tasks = batch_tasks
        self.max_cached_buckets = max_cached_buckets
        self.graph_min_rows, self.graph_warmups = graph_min_rows, graph_warmups
        self.max_squarings, self.adaptive_squarings = max_squarings, bool(adaptive_squarings)
        self.use_graphs, self.profile_execution = bool(use_graphs), bool(profile_execution)
        self.api = self.native_host = self.tight_host = self.native_runtime = self.objective_runtime = None
        self.gradient_buckets = OrderedDict()
        self._pinned_slots = self._pinned_x = self._pinned_result = None
        self._pinned_capacity = self._result_capacity = self.output_capacity = 0
        self._group_capacity = 0
        self._pending = []
        self._orphan_buckets = []
        self._constant_arrays = []
        self.gradient_graph_captures = self.gradient_graph_replays = 0
        self.graph_evictions = self.cache_high_water = 0
        self.requests = self.waves = self.failed_requests = 0
        self.kind_requests = [0, 0]
        self.grouped_row_histogram = [0] * 17
        self.grouped_padded_rows = self.grouped_graph_calls = self.chunk_reductions = 0
        self.parameter_transfer_bytes = self.objective_transfer_bytes = 0
        self.minimum_free_bytes = None
        self.profile = {name: 0. for name in ('validate', 'group', 'plan', 'pack', 'capture', 'enqueue', 'complete', 'scatter')}
        self.source_before = sha(__file__)
        self._host_files = [HOST_DLL, build_record(HOST_DLL)]
        host_metadata = json.loads((build_record(HOST_DLL)).read_text(encoding='utf-8'))
        if sha(HOST_DLL) != host_metadata['dll_sha256']:
            raise ValueError('Original native host DLL differs from its build record')
        for name, expected in host_metadata['source_sha256'].items():
            live = ROOT / 'src/native' / name
            snapshot = snapshot_directory(HOST_DLL) / name
            if sha(live) != expected or sha(snapshot) != expected:
                raise ValueError('Original native host source/snapshot changed: ' + name)
            self._host_files.extend((live, snapshot))
        self.host_before = {str(path.relative_to(ROOT)): sha(path) for path in self._host_files}
        self._runtime_identity_files = [ROOT / 'scripts/cuda_array_round3.py',
            ROOT / 'envs/cuda-torch/Lib/site-packages/torch/include/ATen/native/cuda/Reduce.cuh']
        self.runtime_identity_before = {str(path.relative_to(ROOT)): sha(path) for path in self._runtime_identity_files}
        try:
            self.staging = _Staging()
            # Native host consumes a CPU bounds pointer and does not initialize CUDA.
            PublicationGPULikelihood._prepare_host_bounds(self)
            self.native_host = NativeGraphHost(_CPUHostFactory, self.host_bounds, HOST_DLL)
            d = self.native_host.dll
            d.kaks_publication_validate.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_char_p, C.c_uint32]
            d.kaks_publication_validate.restype = C.c_int
            self.tight_host = TightContext(prepared)
            self.api = NativeCUDAArrayAPI(device=device, expected_uuid=device_uuid or expected_uuid)
            self.torch, self.device = self.api, self.api.device(device)
            self.device_uuid = str(self.api.uuid)
            self._upload_constants()
            self.objective_runtime = NativeObjectiveRuntime(self)
            # Record the object before construction so already created modules
            # remain reachable if a later component constructor fails.
            self.native_runtime = NativeDriverT18Runtime.__new__(NativeDriverT18Runtime)
            self.native_runtime.__init__(self, self.objective_runtime)
            self._free_memory()
            if not _torch_absent():
                raise RuntimeError('Unexpected Torch import while constructing Driver provider')
        except BaseException:
            self._close_resources(constructor_failure=True)
            raise

    @staticmethod
    def _canonical_uuid(value):
        value = str(value).lower().strip()
        return value[4:] if value.startswith('gpu-') else value

    @property
    def peak_allocated_bytes(self):
        return 0 if self.api is None else self.api.peak_allocated_bytes

    def _upload(self, data, shape, dtype):
        result = self.api.frombuffer(data, dtype=dtype).reshape(shape).to(self.device)
        self._constant_arrays.append(result)
        return result

    def _upload_constants(self):
        api = self.api
        pairs = len(self.prepared.values)
        frequencies = array('d')
        for values in self.prepared.values:
            if len(values) < 66:
                raise ValueError('Prepared value row must retain the original64 frequency ABI')
            for model in range(14):
                frequencies.extend([1. / 61] * 64 if model % 2 == 0 else values[2:66])
        self.pi = self._upload(frequencies, (pairs * 14, 64), api.float64)
        self.nkappa = self._upload(array('q', NKAPPA * pairs), (pairs * 14,), api.int64)
        mapping = array('q', (rate for _ in range(pairs) for model in range(14) for rate in RATE_MAP[model // 2]))
        self.rate_map = self._upload(mapping, (pairs * 14, 6), api.int64)
        for name, position in (('edge_i', 0), ('edge_j', 1), ('edge_rate', 2)):
            setattr(self, name, self._upload(array('q', (edge[position] for edge in self.prepared.edges)),
                (len(self.prepared.edges),), api.int64))
        self.edge_syn = self._upload(bytearray(int(edge[3]) for edge in self.prepared.edges),
            (len(self.prepared.edges),), api.bool)
        self.identity = self._upload(array('d', (float(i == j) for i in range(61) for j in range(61))),
            (1, 61, 61), api.float64)

    def _enter(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('CUDA Driver provider requires its creating thread')
        if self._closed or self._quarantined:
            raise RuntimeError('CUDA Driver provider is closed or quarantined after an execution failure')
        if not self._public_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent/reentrant Driver provider call unsupported')

    def _elapsed(self, name, started):
        if self.profile_execution:
            self.profile[name] += time.perf_counter() - started

    def _free_memory(self, enforce_reserve=True):
        free, total = self.api.cuda.mem_get_info(self.device)
        self.minimum_free_bytes = free if self.minimum_free_bytes is None else min(self.minimum_free_bytes, free)
        if enforce_reserve and free < RESERVE:
            raise GraphWorkspaceBudgetError('Driver actual free memory violates2GiB reserve', 1)
        return free, total

    def _drain(self):
        try:
            self.api.synchronize(timeout=10)
        except BaseException:
            # Keep pinned sources, graphs, allocation pools and result references
            # live if completion is unknown. A later explicit close may retry.
            self._quarantined = True
            raise
        self._pending.clear()

    def _ensure_group_outputs(self, count):
        if count > self._group_capacity:
            self._group_capacity = max(count, 64, self._group_capacity * 2)
            n = self._group_capacity
            self.group_order = (C.c_int64 * n)()
            self.group_levels = (C.c_int32 * n)()
            self.split_order = (C.c_int64 * n)()
        self.group_offsets = (C.c_uint32 * 18)()
        self.split_offsets = (C.c_uint32 * 35)()
        if count > self.output_capacity:
            self.output_capacity = max(count, 64, self.output_capacity * 2)
            n = self.output_capacity
            self.output_F = (C.c_double * n)()
            self.output_g = (C.c_double * (n * 7))()
            self.output_diagnostics = (C.c_double * (n * 4))()
            self.output_codes = (C.c_uint32 * n)()

    def _plan(self, count, limit):
        chunks = []
        padded = result_rows = 0
        for budget in range(self.max_squarings + 1):
            for kind in (0, 1):
                begin, end = self.split_offsets[budget * 2 + kind:budget * 2 + kind + 2]
                for offset in range(begin, end, limit):
                    real = min(limit, end - offset)
                    size = max(min(self.graph_min_rows, limit), 1 << (real - 1).bit_length())
                    chunks.append((size, budget, kind, offset, real, padded, result_rows))
                    padded += size
                    result_rows += real
        if result_rows != count:
            raise RuntimeError('Native group/kind split did not cover every request')
        return chunks, padded

    def _ensure_pinned(self, padded, count):
        # Call only before enqueue or after a successful whole-stream drain.
        if self._pending:
            raise RuntimeError('Pinned arenas cannot be rewritten while work is pending')
        if padded > self._pinned_capacity:
            slots = x = None
            try:
                slots = self.api.pinned_host((padded,), self.api.int64)
                x = self.api.pinned_host((padded, 7), self.api.float64)
            except BaseException:
                if slots is not None:
                    slots.close()
                if x is not None:
                    x.close()
                raise
            for previous in (self._pinned_slots, self._pinned_x):
                if previous is not None:
                    previous.close()
            self._pinned_slots, self._pinned_x = slots, x
            self._pinned_capacity = padded
        if count > self._result_capacity:
            result = self.api.pinned_host((count, 11), self.api.float64)
            if self._pinned_result is not None:
                self._pinned_result.close()
            self._pinned_result = result
            self._result_capacity = count

    def _forward_gradient(self, indices, x, budget, kind):
        return self.native_runtime.forward_gradient(indices, x, budget, kind)

    def _close_bucket(self, bucket):
        # Completion must already be known before any allocation is released.
        if bucket.get('graph') is not None:
            bucket['graph'].close()
            bucket['graph'] = None
        if bucket.get('pool') is not None:
            bucket['pool'].close()
            bucket['pool'] = None
        for name in ('indices', 'x'):
            if bucket.get(name) is not None:
                self.api.release(bucket[name])
                bucket[name] = None

    def _evict_oldest(self):
        if not self.gradient_buckets:
            return False
        self._drain()
        key, bucket = next(iter(self.gradient_buckets.items()))
        self._close_bucket(bucket)
        del self.gradient_buckets[key]
        self.graph_evictions += 1
        return True

    def _new_gradient_bucket(self, size, budget, kind, input_row):
        estimate = size * 61 * 61 * 8 * (256 + 4 * budget) + 64 * 1024**2
        while len(self.gradient_buckets) >= self.max_cached_buckets:
            self._evict_oldest()
        while estimate + RESERVE > self._free_memory(enforce_reserve=False)[0]:
            if not self._evict_oldest():
                raise GraphWorkspaceBudgetError('Driver pool estimate plus2GiB reserve exceeds free memory', max(1, size // 2))
        bucket = {'graph': None, 'pool': None, 'indices': None, 'x': None, 'result': None}
        capture_started = time.perf_counter()
        try:
            bucket['indices'] = self.api.empty((size,), dtype=self.api.int64, device=self.device)
            bucket['x'] = self.api.empty((size, 7), dtype=self.api.float64, device=self.device)
            self._h2d(bucket, size, input_row)
            # One real dry evaluation records/owns every allocation in the tape.
            # Capture replay must request exactly this sequence of shape/dtypes.
            self.api.begin_record()
            try:
                bucket['result'] = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
            finally:
                bucket['pool'] = self.api.end_record()
            self._drain()
            for _ in range(self.graph_warmups - 1):
                with self.api.capture_pool(bucket['pool']):
                    bucket['result'] = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
                self._drain()
            if self.use_graphs:
                with self.api.capture_pool(bucket['pool']):
                    self.api.capture_begin()
                    try:
                        bucket['result'] = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
                        bucket['graph'] = self.api.capture_end()
                    except BaseException:
                        self.api.abort_capture()
                        raise
                self.gradient_graph_captures += 1
            self._free_memory()
        except BaseException:
            try:
                self._drain()
                self._close_bucket(bucket)
            except BaseException:
                # Preserve partially created graphs/pools/inputs for a later
                # explicit cleanup retry; never unload their kernel modules.
                self._orphan_buckets.append(bucket)
                raise
            raise
        self.gradient_buckets[(size, budget, kind)] = bucket
        self.cache_high_water = max(self.cache_high_water, len(self.gradient_buckets))
        self._elapsed('capture', capture_started)
        return bucket

    def _h2d(self, bucket, size, input_row):
        self.api.copy_h2d(bucket['indices'], self._pinned_slots, nbytes=size * 8, host_offset=input_row * 8)
        self.api.copy_h2d(bucket['x'], self._pinned_x, nbytes=size * 56, host_offset=input_row * 56)
        self.parameter_transfer_bytes += size * 64

    def _evaluate_wave(self, slots, parameters, kinds, count):
        self._ensure_group_outputs(count)
        started = time.perf_counter()
        self.native_host._check(self.native_host.dll.kaks_publication_validate(self.native_host.context,
            slots, parameters, count, self.native_host.error, len(self.native_host.error)))
        self._elapsed('validate', started)
        started = time.perf_counter()
        self.native_host._check(self.tight_host.group(self.native_host.context, slots, parameters, count,
            self.max_squarings, int(self.adaptive_squarings), self.group_order, self.group_levels,
            self.group_offsets, C.byref(self.native_host.maximum), self.native_host.error, len(self.native_host.error)))
        self.staging.check(self.staging.dll.drv_split(self.group_order, count, kinds, self.group_offsets,
            self.split_order, self.split_offsets), 'stable kind split')
        self._elapsed('group', started)
        limit = self.batch_tasks
        while True:
            started = time.perf_counter()
            chunks, padded = self._plan(count, limit)
            self._ensure_pinned(padded, count)
            self._elapsed('plan', started)
            started = time.perf_counter()
            for size, budget, kind, offset, real, input_row, output_row in chunks:
                self.staging.check(self.staging.dll.drv_pack(slots, parameters, self.split_order, count,
                    offset, real, size, self._pinned_slots.data_ptr() + input_row * 8,
                    self._pinned_x.data_ptr() + input_row * 56), 'request pack')
            self._elapsed('pack', started)
            try:
                started = time.perf_counter()
                for size, budget, kind, offset, real, input_row, output_row in chunks:
                    key = (size, budget, kind)
                    bucket = self.gradient_buckets.get(key)
                    if bucket is None:
                        bucket = self._new_gradient_bucket(size, budget, kind, input_row)
                    else:
                        self.gradient_buckets.move_to_end(key)
                        self._h2d(bucket, size, input_row)
                    if bucket['graph'] is not None:
                        bucket['graph'].launch()
                        self.gradient_graph_replays += 1
                        result = bucket['result']
                    else:
                        with self.api.capture_pool(bucket['pool']):
                            result = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
                    self.api.copy_d2h(self._pinned_result, result, nbytes=real * 11 * 8,
                        host_offset=output_row * 11 * 8)
                    self._pending.append((bucket, result))
                    self.grouped_padded_rows += size
                    self.grouped_graph_calls += 1
                    self.objective_transfer_bytes += real * 11 * 8
                self._elapsed('enqueue', started)
                started = time.perf_counter()
                self._drain()
                self._elapsed('complete', started)
                self._free_memory()
                break
            except GraphWorkspaceBudgetError as error:
                self._drain()
                if limit == 1:
                    raise
                limit = min(limit // 2, max(1, error.suggested_rows))
                limit = 1 << (limit.bit_length() - 1)
                self.chunk_reductions += 1
            except BaseException:
                # Even immediate launch failure can follow earlier queued copies.
                self._drain()
                self._quarantined = True
                raise
        started = time.perf_counter()
        for size, budget, kind, offset, real, input_row, output_row in chunks:
            self.staging.check(self.staging.dll.drv_scatter(self.split_order, count, offset, real,
                self._pinned_result.data_ptr() + output_row * 11 * 8,
                self.output_F, self.output_g, self.output_diagnostics, self.output_codes), 'result scatter')
        self._elapsed('scatter', started)
        self.requests += count
        self.waves += 1
        for budget in range(self.max_squarings + 1):
            self.grouped_row_histogram[budget] += self.group_offsets[budget + 1] - self.group_offsets[budget]
            for kind in (0, 1):
                self.kind_requests[kind] += self.split_offsets[budget * 2 + kind + 1] - self.split_offsets[budget * 2 + kind]
        f = (C.c_double * count).from_buffer(self.output_F)
        g = (C.c_double * (count * 7)).from_buffer(self.output_g)
        self.last_diagnostics = (C.c_double * (count * 4)).from_buffer(self.output_diagnostics)
        self.last_codes = (C.c_uint32 * count).from_buffer(self.output_codes)
        bad = sum(self.last_codes)
        self.failed_requests += bad
        if bad:
            raise GPUGradientError('CUDA Driver FP64 rows failed the unchanged finite/scaling guard; no clipping or truncation',
                objectives=list(f), gradient_log=[list(g[row * 7:row * 7 + 7]) for row in range(count)],
                diagnostics=[list(self.last_diagnostics[row * 4:row * 4 + 4]) for row in range(count)],
                codes=list(self.last_codes))
        return f, g

    def mixed_buffered(self, slots, parameters, kinds, count=None, diagnostics=False):
        self._enter()
        try:
            if not isinstance(slots, C.Array) or slots._type_ is not C.c_int64:
                raise ValueError('Expected ctypes int64 slots')
            if not isinstance(parameters, C.Array) or parameters._type_ is not C.c_double:
                raise ValueError('Expected ctypes FP64 x7')
            if not isinstance(kinds, C.Array) or kinds._type_ is not C.c_uint32:
                raise ValueError('Expected ctypes uint32 kinds')
            count = len(slots) if count is None else count
            if type(count) is not int or not 0 <= count <= MAX_REQUESTS or count > min(len(slots), len(kinds)) or count * 7 > len(parameters):
                raise ValueError('Invalid borrowed count/capacity; maximum1e6 requests per synchronous wave')
            if not count:
                return (C.c_double * 0)(), (C.c_double * 0)()
            return self._evaluate_wave(slots, parameters, kinds, count)
        finally:
            self._public_lock.release()

    def objective_gradient(self, slots, parameters, count=None, diagnostics=False):
        count = len(slots) if count is None else count
        if type(count) is not int or not 0 <= count <= len(slots):
            raise ValueError('Invalid objective-gradient count')
        return self.mixed_buffered(slots, parameters, (C.c_uint32 * count)(*([1] * count)), count, diagnostics)

    def objectives(self, slots, parameters, count=None):
        count = len(slots) if count is None else count
        if type(count) is not int or not 0 <= count <= len(slots):
            raise ValueError('Invalid objective count')
        return self.mixed_buffered(slots, parameters, (C.c_uint32 * count)(), count)[0]

    def graph_manifest(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Manifest requires the creating thread')
        if self._closed:
            raise RuntimeError('Manifest must be captured before provider close')
        host_after = {str(path.relative_to(ROOT)): sha(path) for path in self._host_files}
        runtime_identity_after = {str(path.relative_to(ROOT)): sha(path) for path in self._runtime_identity_files}
        if (sha(__file__) != self.source_before or host_after != self.host_before
                or runtime_identity_after != self.runtime_identity_before or not _torch_absent()):
            raise RuntimeError('Driver source/native host identity changed or Torch imported')
        runtime = self.native_runtime.manifest()
        runtime['Torch_scope'] = 'None; explicit NativeCUDAArrayAPI owns allocations, views, currentstream and Driver graphs'
        return {'schema': 'kaksforge.native_driver_gradient_round3/1',
            'device_uuid': self.device_uuid, 'driver': self.api.manifest(), 'Torch_imported': False,
            'provider_source_sha256': self.source_before, 'provider_source_unchanged': True,
            'native_host_before': self.host_before, 'native_host_after': host_after,
            'array_compiler_builtins_header_before': self.runtime_identity_before,
            'array_compiler_builtins_header_after': runtime_identity_after,
            'tight_host': self.tight_host.manifest(), 'staging': self.staging.manifest(),
            'native_math': runtime, 'full_native_forward_gradient_inherited_unchanged': True,
            'sparse_constructor_only_change': 'Omit unused Torch AD decorator creation; all products inherited',
            'gradient_coordinate': 'Raw F and x*dF/draw_x; no division by codon L; inactive0',
            'precision': 'Existing FP64 PTX only; no fastmath/TF32/precision change',
            'request_order': 'Stable native budget/kind grouping; complete original-ordinal scatter before optimizer receives wave',
            'padding': 'Power-of-two floor rows; exact first-real request duplicated, only real outputs consumed',
            'capture_allocation': 'One actual dry tape records allocation sequence; pool replay requires exact sequence; no malloc/free in capture',
            'wave_completion': 'All H2D/launch/D2H on one owned stream; one terminal synchronization, plus explicit coldsetup/LRU drains',
            'numeric_failure': 'Original actual norm/required scaling and finite F/g; partial NaN/code1, no clamp/truncation',
            'batch_tasks': self.batch_tasks, 'max_cached_buckets': self.max_cached_buckets,
            'graph_min_rows': self.graph_min_rows, 'graph_warmups': self.graph_warmups,
            'max_squarings': self.max_squarings, 'adaptive_squarings': self.adaptive_squarings, 'use_graphs': self.use_graphs,
            'gradient_bucket_keys': [list(key) for key in self.gradient_buckets],
            'gradient_graph_captures': self.gradient_graph_captures, 'gradient_graph_replays': self.gradient_graph_replays,
            'graph_evictions': self.graph_evictions, 'cache_high_water': self.cache_high_water,
            'requests': self.requests, 'waves': self.waves, 'kind_requests': self.kind_requests,
            'failed_requests': self.failed_requests, 'grouped_row_histogram': self.grouped_row_histogram,
            'grouped_padded_rows': self.grouped_padded_rows, 'grouped_graph_calls': self.grouped_graph_calls,
            'parameter_transfer_bytes': self.parameter_transfer_bytes, 'objective_transfer_bytes': self.objective_transfer_bytes,
            'transfer_and_chunk_counters_include_coldsetup_and_retry_attempts': True,
            'chunk_reductions': self.chunk_reductions, 'peak_allocated_bytes': self.peak_allocated_bytes,
            'minimum_free_bytes': self.minimum_free_bytes, 'reserve_bytes': RESERVE,
            'profile_execution': self.profile_execution, 'CPU_profile_seconds': self.profile if self.profile_execution else None,
            'profile_scope': 'CPU wall segments; enqueue/capture include overlap/drains and must not be added to GPU event timings',
            'optimizer_called': False, 'requires_original_HP28_and_full_development_gates': True}

    def release_graph_cache(self):
        self._enter()
        try:
            self._drain()
            while self.gradient_buckets:
                self._evict_oldest()
        finally:
            self._public_lock.release()

    def _close_resources(self, constructor_failure=False):
        try:
            if self.api is not None:
                self._drain()
                for key, bucket in list(self.gradient_buckets.items()):
                    self._close_bucket(bucket)
                    del self.gradient_buckets[key]
                while self._orphan_buckets:
                    self._close_bucket(self._orphan_buckets[-1])
                    self._orphan_buckets.pop()
                for name in ('_pinned_slots', '_pinned_x', '_pinned_result'):
                    allocation = getattr(self, name, None)
                    if allocation is not None:
                        allocation.close()
                        setattr(self, name, None)
                components = []
                if self.native_runtime is not None:
                    for name in ('matrix', 'gemm', 'sparse', 'coefficients', 'helpers'):
                        component = getattr(self.native_runtime, name, None)
                        if component is not None:
                            components.append(component)
                            if name == 'matrix' and hasattr(component, 'reverse'):
                                components.append(component.reverse)
                if self.objective_runtime is not None:
                    components.append(self.objective_runtime)
                for component in components:
                    module = getattr(component, 'module', None)
                    if module is not None and module.value:
                        function = component.driver.cuModuleUnload
                        function.argtypes, function.restype = [C.c_void_p], C.c_int
                        code = function(module)
                        if code:
                            raise RuntimeError('Owned Driver module cleanup failed: ' + str(code))
                        component.module = C.c_void_p()
                # Context destruction also releases a module created before its
                # failed constructor could expose the component to this owner.
                self.api.close()
        finally:
            # Independent host contexts are safe to release even when a device
            # timeout requires preservation of GPU/pinned resources.
            try:
                if self.tight_host is not None:
                    self.tight_host.close()
            finally:
                if self.native_host is not None:
                    self.native_host.close()
        self._closed = True

    def close(self):
        if self._closed:
            return
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Driver close requires the creating thread')
        if not self._public_lock.acquire(blocking=False):
            raise RuntimeError('Driver close cannot overlap a provider call')
        try:
            self._close_resources()
        finally:
            self._public_lock.release()

    def __enter__(self):
        self._enter()
        self._public_lock.release()
        return self

    def __exit__(self, *_):
        self.close()
