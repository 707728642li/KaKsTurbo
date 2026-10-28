"""FP64 structural sparse products; full first-order adjoints remain dense.

Only Z is structural: the code1 graph has263 undirected edges and61 diagonal
positions. Dense operands and gradients must not be support-masked. CUDA is
initialized only by SparseRuntime; build_sparse_ptx compiles with NVRTC on CPU.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha

import ctypes as C
import hashlib
import json
from pathlib import Path

from cuda_driver import compile_ptx
from torch_matrix_prototype import codon_edges

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/gpu_round2/sparse_t18_v1.cu'
BUILD = ROOT / 'build/gpu-sparse-t18-round2-fp64-v1'
OPTIONS = ['--std=c++17', '--gpu-architecture=compute_86', '--fmad=false']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structural_csr(edges=None):
    canonical = codon_edges()
    if edges is not None and [tuple(edge) for edge in edges] != canonical:
        raise ValueError('Sparse Z requires the exact code1 codon_edges contract')
    columns = [{row} for row in range(61)]
    for i, j, _, _ in canonical:
        columns[i].add(j); columns[j].add(i)
    pointers, indices = [0], []
    for row in columns:
        indices.extend(sorted(row)); pointers.append(len(indices))
    if len(canonical) != 263 or len(indices) != 587:
        raise RuntimeError('Unexpected original code1 structural sparsity')
    return pointers, indices


def kernel_source():
    pointers, indices = structural_csr()
    template = SOURCE.read_text(encoding='utf-8')
    if template.count('/*ROW_POINTERS*/') != 1 or template.count('/*COLUMN_INDICES*/') != 1:
        raise RuntimeError('Sparse kernel template markers changed')
    return (template.replace('/*ROW_POINTERS*/', ','.join(map(str, pointers)))
            .replace('/*COLUMN_INDICES*/', ','.join(map(str, indices))))


def build_sparse_ptx():
    """Compile/load immutable project PTX; no CUDA context/device operation."""
    compiler = ROOT / 'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins = ROOT / 'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    pointers, indices = structural_csr()
    csr = {'N': 61, 'undirected_edges': 263, 'entries': 587,
           'row_pointers': pointers, 'column_indices': indices,
           'contract': 'Exact codon_edges(), diagonal plus both directions, ascending k'}
    csr_bytes = (json.dumps(csr, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
    source = kernel_source()
    identity = {'template_sha256': sha(SOURCE),
                'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
                'CSR_sha256': hashlib.sha256(csr_bytes).hexdigest(),
                'options': OPTIONS, 'nvrtc_sha256': historical_compiler_sha(compiler), 'nvrtc_builtins_sha256': historical_compiler_sha(builtins)}
    BUILD.mkdir(parents=True, exist_ok=True)
    metadata_path, ptx_path = BUILD / 'build.json', BUILD / 'sparse_t18.ptx'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise RuntimeError('Preserve earlier sparse build: source/CSR/options/compiler identity changed')
        ptx = ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest() != metadata['ptx_sha256']:
            raise RuntimeError('Sparse PTX identity changed')
    else:
        if any(BUILD.iterdir()):
            raise RuntimeError('Preserve incomplete earlier sparse build; select a new version')
        try:
            ptx, log = compile_ptx(source, OPTIONS)
        except BaseException as error:
            (BUILD / 'compile_failure.json').write_text(json.dumps(dict(identity,
                error_type=type(error).__name__, error=str(error), GPU_context_created=False), indent=2) + '\n', encoding='utf-8')
            raise
        ptx_path.write_bytes(ptx)
        (BUILD / 'sparse_t18.cu').write_bytes(source.encode('utf-8'))
        (BUILD / 'sparse_t18_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD / 'structural_csr.json').write_bytes(csr_bytes)
        (BUILD / 'build.log').write_text(log, encoding='utf-8')
        metadata = dict(identity, schema='kaksforge.sparse_t18_FP64_build_round2/1',
            ptx_sha256=hashlib.sha256(ptx).hexdigest(), precision='FP64 inputs/output/accumulation',
            arithmetic='Explicit double __fma_rn in ascending retained k; only structural zero primal terms skipped',
            undirected_edges=263, CSR_entries=587, CUDA_context_created_by_compilation=False,
            fast_math=False, dense_gradients_support_masked=False)
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    if ((BUILD / 'sparse_t18.cu').read_bytes() != source.encode('utf-8')
            or (BUILD / 'sparse_t18_template.cu').read_bytes() != SOURCE.read_bytes()
            or (BUILD / 'structural_csr.json').read_bytes() != csr_bytes):
        raise RuntimeError('Sparse compiled source/template/CSR copy changed')
    return ptx, metadata


def make_sparse_products(torch, runtime):
    """Same first-order rules for CUDA kernels and the separately run CPU gate."""
    from torch.autograd.function import once_differentiable

    class SparseBoth(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return runtime.product_left(a, b)
        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            a, b = ctx.saved_tensors
            # Both primal operands have structural support, but both
            # adjoints are FULL dense matrices, including non-edge entries.
            ga = runtime.product_right(incoming, b.transpose(-1, -2)) if ctx.needs_input_grad[0] else None
            gb = runtime.product_left(a.transpose(-1, -2), incoming) if ctx.needs_input_grad[1] else None
            return ga, gb

    class SparseLeft(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return runtime.product_left(a, b)
        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            a, b = ctx.saved_tensors
            # a=Z and b=A2(dense): dZ remains the original FULL dense GEMM.
            ga = runtime.dense.product(incoming, b.transpose(-1, -2)) if ctx.needs_input_grad[0] else None
            gb = runtime.product_left(a.transpose(-1, -2), incoming) if ctx.needs_input_grad[1] else None
            return ga, gb

    return SparseBoth.apply, SparseLeft.apply


class SparseRuntime:
    def __init__(self, torch, device, dense_runtime, edges=None):
        self.torch, self.device, self.dense = torch, torch.device(device), dense_runtime
        if self.device.type != 'cuda' or self.device.index is None:
            raise ValueError('SparseRuntime requires an explicit selected CUDA device')
        structural_csr(edges)
        ptx, self.metadata = build_sparse_ptx()
        self.path = BUILD
        self.artifacts = {name: sha(BUILD / name) for name in
                          ('build.json', 'sparse_t18.cu', 'sparse_t18_template.cu', 'structural_csr.json', 'sparse_t18.ptx')}
        self.driver = cuda_driver(); self.module = C.c_void_p()
        self._bind('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p])
        self._bind('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
        self._bind('cuLaunchKernel', [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.c_void_p, C.c_void_p])
        self._bind('cuModuleUnload', [C.c_void_p])
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device)
            blob = C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module), blob))
            self.functions = {}
            try:
                for name in ('sparse_left', 'sparse_right'):
                    function = C.c_void_p()
                    self._check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()))
                    self.functions[name] = function
            except BaseException as error:
                # No launch/graph can exist during partial constructor failure.
                cleanup_code = self.driver.cuModuleUnload(self.module)
                if cleanup_code:
                    raise RuntimeError('Sparse module function lookup failed; partial-module unload also failed: ' + str(cleanup_code)) from error
                self.module = C.c_void_p()
                raise
        self.square, self.left_matmul = make_sparse_products(torch, self)

    def _bind(self, name, arguments):
        function = getattr(self.driver, name); function.argtypes = arguments; function.restype = C.c_int

    def _check(self, code):
        if code:
            raise RuntimeError('CUDA sparse driver operation failed: ' + str(code))

    def _product(self, name, sparse, dense):
        t = self.torch
        if (sparse.ndim != 3 or dense.shape != sparse.shape or sparse.shape[-2:] != (61, 61)
                or not 1 <= sparse.shape[0] <= 65535 or sparse.dtype != t.float64 or dense.dtype != t.float64
                or sparse.device != self.device or dense.device != self.device
                or any(stride < 0 for stride in (*sparse.stride(), *dense.stride()))):
            raise ValueError('Sparse product requires same-shaped FP64 CUDA61x61 tensors,1..65535rows,nonnegative strides')
        out = t.empty(sparse.shape, dtype=t.float64, device=self.device)
        arguments = [C.c_uint64(tensor.data_ptr()) for tensor in (sparse, dense, out)]
        arguments.extend(C.c_int64(stride) for stride in (*sparse.stride(), *dense.stride()))
        arguments.append(C.c_int(sparse.shape[0]))
        pointers = (C.c_void_p * len(arguments))(*(C.cast(C.byref(value), C.c_void_p) for value in arguments))
        with t.cuda.device(self.device):
            stream = t.cuda.current_stream(self.device).cuda_stream
            self._check(self.driver.cuLaunchKernel(self.functions[name], 2, 16, sparse.shape[0], 32, 4, 1, 0, stream, pointers, None))
        return out

    def product_left(self, sparse, dense):
        return self._product('sparse_left', sparse, dense)

    def product_right(self, dense, sparse):
        return self._product('sparse_right', sparse, dense)

    def manifest(self):
        after = {name: sha(self.path / name) for name in self.artifacts}
        if after != self.artifacts or sha(SOURCE) != self.metadata['template_sha256']:
            raise RuntimeError('Sparse source/build artifact changed during execution')
        return dict(self.metadata, build=str(self.path),
            product_scope='Z*Z and Z*A2 only; all other primal products dense',
            reverse='Z*Z: full dA=G*Z^T sparse-right,dB=Z^T*G sparse-left; Z*A2: full dZ dense,dA2 sparse-left',
            structural_support='587positions,263edges in bothdirections+61diag; no numerical-value cutoff or adjoint mask',
            storage='All output and returned gradients remain dense(batch,61,61); transposed/nonnegative-stride input reads',
            artifact_SHA256_before=self.artifacts, artifact_SHA256_after=after, artifact_unchanged=True,
            CUDA_module_lifetime='Owned by provider/graphs until isolated process exit; no unsafe destructor',
            higher_order_gradients=False, precision_change=False, gradient_bit_parity_claimed=False,
            requires_original_HP28_and_complete_development_fit_gates=True)
