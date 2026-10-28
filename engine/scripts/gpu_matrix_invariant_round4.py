"""Isolated row-independent FP64 mean-rate addition schedule for round4.

Forward API is inherited from the frozen ForwardMatrixFusionRuntime.  Only
the old width/row-address emulation of Torch's 263-edge reduction is removed:
all requests use exactly the old row0/width32 schedule.  Full unmasked mean
rate, strict-positive support, all263 edges, dense emissions, and the existing
fixed-tree analytic reverse remain. Import and --build never initialize CUDA.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import threading

from cuda_driver import compile_ptx
from gpu_forward_matrix_fusion_round2 import ForwardMatrixFusionRuntime
from gpu_matrix_contract_round2 import MatrixContractRuntime
from torch_matrix_prototype import codon_edges

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/gpu_round4/matrix_numerical_invariant_v1.cu'
ORIGINAL = ROOT / 'src/gpu_round2/matrix_forward_structural_zero_v1.cu'
REVERSE_SOURCE = ROOT / 'src/gpu_round2/matrix_contract_v1.cu'
BUILD = ROOT / 'build/gpu-matrix-invariant-round4-fp64-v1'
OPTIONS = ['--std=c++17', '--gpu-architecture=compute_86', '--fmad=false']
RUNTIME_SOURCE = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_source():
    edges = codon_edges()
    if len(edges) != 263:
        raise ValueError('The frozen code1 contract requires all263 edges')
    codes = [-1] * 3721
    for i, j, rate, synonymous in edges:
        codes[i * 61 + j] = codes[j * 61 + i] = rate + (0 if synonymous else 6)
    replacements = {
        'EDGE_CODE_VALUES': codes, 'EDGE_I_VALUES': [e[0] for e in edges],
        'EDGE_J_VALUES': [e[1] for e in edges], 'EDGE_R_VALUES': [e[2] for e in edges],
        'EDGE_SYN_VALUES': [int(e[3]) for e in edges],
    }
    text = SOURCE.read_text(encoding='utf-8')
    for key, values in replacements.items():
        text = text.replace(key, ','.join(map(str, values)))
    return text


def static_contract():
    """Verify that numerical changes are exactly the two scheduling constants."""
    old = ORIGINAL.read_text(encoding='utf-8')
    expected = old.replace(
        "// Reduction schedule follows installed Torch Reduce.cuh's contiguous 61/263\n"
        '// layouts for power-of-two graph rows. No bitwise equivalence claim is made.',
        '// Round4 uses the row0/x32 mean-rate addition schedule for EVERY request.\n'
        '// The schedule is independent of row ordinal, graph bucket, and partition.\n'
        '// Real-number formulas and every per-edge FP64 multiply remain unchanged.')
    expected = expected.replace(
        '// 263-wide contiguous sum uses input vectorization4. Torch chooses\n'
        "    // block_x32 when num_outputs>=16, else64. Each row's 32-byte alignment\n"
        '    // shift is (row*263)%4 since the original tensors start aligned.\n'
        '    int width=rows>=16?32:64;',
        '// Fixed 263-edge mean-rate reduction. Preserve old row0/x32 operation\n'
        '    // order for EVERY row: no dependence on row count, ordinal, or address.\n'
        '    // This deliberately removes the old Torch alignment-emulation schedule.\n'
        '    const int width=32;')
    expected = expected.replace('int shift=(row*263)%4,head=shift?4-shift:0;',
        'const int shift=0,head=0;')
    live = SOURCE.read_text(encoding='utf-8')
    if live != expected:
        raise RuntimeError('Round4 forward source has changes beyond preregistered reduction scheduling')
    return dict(passed=True, numerical_changes=['width=32 for every row count',
        'shift=0 and head=0 for every row ordinal'],
        all_other_source_statements_unchanged=True,
        original_forward_template_sha256=sha(ORIGINAL),
        new_forward_template_sha256=sha(SOURCE),
        existing_reverse_source_sha256=sha(REVERSE_SOURCE),
        full_unmasked_mean_rate=True, all263_edges=True, FP64=True,
        formula_optimizer_bounds_convergence_models_unchanged=True,
        partition_or_row_bit_parity_requires_GPU_gate=True)


def build_invariant_matrix_ptx():
    contract = static_contract()
    generated = kernel_source()
    compiler = ROOT / 'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins = ROOT / 'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    identity = dict(source_template_sha256=sha(SOURCE),
        generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        original_template_sha256=sha(ORIGINAL), reverse_source_sha256=sha(REVERSE_SOURCE),
        options=OPTIONS, nvrtc_sha256=historical_compiler_sha(compiler), nvrtc_builtins_sha256=historical_compiler_sha(builtins))
    BUILD.mkdir(parents=True, exist_ok=True)
    metadata_path, ptx_path = BUILD / 'build.json', BUILD / 'matrix_invariant.ptx'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise RuntimeError('Preserve prior invariant build; source/compiler/options changed')
        ptx = ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest() != metadata['ptx_sha256']:
            raise RuntimeError('Invariant matrix PTX changed')
    else:
        if any(BUILD.iterdir()):
            raise RuntimeError('Preserve incomplete invariant build; choose a new version')
        try:
            ptx, log = compile_ptx(generated, OPTIONS)
        except BaseException as error:
            (BUILD / 'compile_failure.json').write_text(json.dumps(dict(identity,
                error_type=type(error).__name__, error=str(error), CUDA_context_created=False),
                indent=2) + '\n', encoding='utf-8')
            raise
        for path, key in ((SOURCE, 'source_template_sha256'), (ORIGINAL, 'original_template_sha256'),
                (REVERSE_SOURCE, 'reverse_source_sha256'), (compiler, 'nvrtc_sha256'),
                (builtins, 'nvrtc_builtins_sha256')):
            if sha(path) != identity[key]:
                raise RuntimeError('Invariant source/compiler changed during CPU compilation')
        ptx_path.write_bytes(ptx)
        (BUILD / 'matrix_invariant.cu').write_bytes(generated.encode())
        (BUILD / 'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD / 'build.log').write_text(log, encoding='utf-8')
        metadata = dict(identity, schema='kaksforge.matrix_row_invariant_FP64_build_round4/1',
            ptx_sha256=hashlib.sha256(ptx).hexdigest(), static_contract=contract,
            precision='FP64; no fastmath or TF32; fmad=false',
            reduction='263 edges fixed row0/x32/vector4 schedule independent of rows and ordinal',
            reverse='Existing fixed-tree MatrixContractRuntime; no changes or masking',
            CUDA_context_created_by_compilation=False, optimizer_or_MA_changed=False,
            module_lifetime='process-owned', actual_GPU_bitparity_and_HP28_required=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    if (BUILD / 'matrix_invariant.cu').read_bytes() != generated.encode() \
            or (BUILD / 'source_template.cu').read_bytes() != SOURCE.read_bytes():
        raise RuntimeError('Invariant compiled source copies changed')
    return ptx, metadata


class InvariantForwardMatrixRuntime(ForwardMatrixFusionRuntime):
    """Drop-in native matrix stage; untouched forward API and analytic reverse."""
    def __init__(self, owner, reverse=None):
        self.owner, self.torch, self.device = owner, owner.torch, owner.device
        if [tuple(edge) for edge in owner.prepared.edges] != codon_edges():
            raise ValueError('Invariant matrix requires the exact code1 edges')
        self.reverse = MatrixContractRuntime(owner) if reverse is None else reverse
        if getattr(self.reverse, 'owner', None) is not owner:
            raise ValueError('Reverse matrix component belongs to another owner')
        ptx, self.metadata = build_invariant_matrix_ptx()
        self.path = BUILD
        self.runtime_source_sha256 = sha(RUNTIME_SOURCE)
        self.artifacts = {name: sha(BUILD / name) for name in
            ('build.json', 'matrix_invariant.cu', 'source_template.cu', 'matrix_invariant.ptx')}
        self.driver, self.module = cuda_driver(), C.c_void_p()
        self._bind('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p])
        self._bind('cuModuleUnload', [C.c_void_p])
        self._bind('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
        self._bind('cuLaunchKernel', [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.c_void_p, C.c_void_p])
        with self.torch.cuda.device(self.device):
            self.torch.cuda.current_stream(self.device)
            blob = C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module), blob))
            self.functions = {}
            try:
                for name in ('matrix_forward_prepare', 'matrix_forward_dense'):
                    function = C.c_void_p()
                    self._check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()))
                    self.functions[name] = function
            except BaseException as error:
                code = self.driver.cuModuleUnload(self.module)
                if code:
                    raise RuntimeError('Invariant module lookup and partial unload failed: ' + str(code)) from error
                self.module = C.c_void_p()
                raise

    def manifest(self):
        after = {name: sha(self.path / name) for name in self.artifacts}
        static_contract()
        if after != self.artifacts or sha(RUNTIME_SOURCE) != self.runtime_source_sha256:
            raise RuntimeError('Invariant runtime/artifacts/source changed during execution')
        return dict(self.metadata, build=str(self.path), reverse=self.reverse.manifest(),
            runtime_source_sha256=self.runtime_source_sha256, runtime_source_unchanged=True,
            input_output='Existing A/pi/sqrt/live/S/mr/rates and native analytic reverse API',
            artifact_SHA256_before=self.artifacts, artifact_SHA256_after=after, artifact_unchanged=True,
            requires_HP28_and_full_partition_fit_gates=True)


def install_invariant_matrix(provider):
    """Replace constructor's forward module before any request or graph capture.

    Reuse the existing same-owner reverse module. If unloading the old forward
    fails, leave it installed and release the newly constructed forward module.
    The provider continues to own, drain, and release every surviving module.
    """
    if threading.get_ident() != provider.owner_thread:
        raise RuntimeError('Invariant matrix installation requires the creating thread')
    if provider.requests or provider.waves or provider.gradient_buckets:
        raise RuntimeError('Invariant matrix must be installed before requests and graph capture')
    provider._enter()
    try:
        old = provider.native_runtime.matrix
        if isinstance(old, InvariantForwardMatrixRuntime):
            raise RuntimeError('Invariant matrix already installed')
        with provider.torch.cuda.device(provider.device):
            new = InvariantForwardMatrixRuntime(provider, reverse=old.reverse)
            code = old.driver.cuModuleUnload(old.module)
            if code:
                cleanup = new.driver.cuModuleUnload(new.module)
                if cleanup:
                    raise RuntimeError('Old and new matrix modules could not unload: ' + str((code, cleanup)))
                new.module = C.c_void_p()
                raise RuntimeError('Original forward module could not unload: ' + str(code))
            old.module = C.c_void_p()
            provider.native_runtime.matrix = new
            return new
    finally:
        provider._public_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        ptx, metadata = build_invariant_matrix_ptx()
        print(json.dumps(dict(build=str(BUILD), ptx_bytes=len(ptx),
            ptx_sha256=metadata['ptx_sha256'], CUDA_context_created=False, static_contract=static_contract())))
    else:
        print(json.dumps(static_contract()))


if __name__ == '__main__':
    main()
