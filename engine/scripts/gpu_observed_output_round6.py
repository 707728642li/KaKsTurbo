"""Exact observable final-forward GEMM cells with full dense output storage.

Scope is budget0 final D@C only: every prepared observed cell executes the
original64 FP64 FMA sequence. Unobserved product cells are positive zero and
must never be consumed by a later matrix product or norm. No GPU work occurs
on import; cached compilation is explicitly CPU NVRTC only.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

from cuda_driver import compile_ptx
from gpu_observed_adjoint_round6 import ObservedAdjointRuntime,OPTIONS,ORIGINAL_SOURCE
from gpu_register_gemm_round5 import validate_operands

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round6/observed_output_product_v3.cu'
RUNTIME_SOURCE=Path(__file__).resolve()
BUILD=ROOT/'build/gpu-observed-output-round6-fp64-v3'
ARTIFACTS=('build.json','source_template.cu','nvrtc_input.cu','observed_output.ptx')
OBJECTIVE_SOURCE=ROOT/'src/gpu_round2/native_objective_v1.cu'
HELPER_SOURCE=ROOT/'src/gpu_round2/native_t18_intscale_helpers_v1.cu'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build_observed_output(allow_compile=False):
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    source=SOURCE.read_text(encoding='utf-8')
    identity=dict(source_sha256=sha(SOURCE),NVRTC_input_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_source_sha256=sha(RUNTIME_SOURCE),original_source_sha256=sha(ORIGINAL_SOURCE),
        objective_source_sha256=sha(OBJECTIVE_SOURCE),helper_source_sha256=sha(HELPER_SOURCE),
        options=OPTIONS,nvrtc_sha256=historical_compiler_sha(compiler),nvrtc_builtins_sha256=historical_compiler_sha(builtins))
    BUILD.mkdir(parents=True,exist_ok=True);record=BUILD/'build.json';ptx_path=BUILD/'observed_output.ptx'
    if record.exists():
        metadata=json.loads(record.read_text())
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve observed-output v3 build; identity changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Observed-output v3 PTX changed')
    else:
        if not allow_compile:raise RuntimeError('Cached observed-output PTX required; use explicit CPU-only --build')
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete observed-output v3 build')
        ptx,log=compile_ptx(source,OPTIONS)
        ptx_path.write_bytes(ptx);(BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'nvrtc_input.cu').write_bytes(source.encode());(BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            schema='kaksforge.observed_output_product_build_round6/3',
            precision='FP64 input/output/accumulator; no fast-math/TF32',
            arithmetic='Selected outputs original serial k0..63 __fma_rn and positive-zero padding; no k truncation or reassociation',
            output_scope='Budget0 LAST forward D@C only; prepared weight>=1e-6 output cells; all input states retained',
            layout='One256-thread CTA per request; fused full zero fill, barrier, packed CSR output scatter; complete right operand shared FP64[61,65]',
            CUDA_context_created_by_compilation=False,optimizer_or_MA_changed=False,precision_change=False)
        record.write_text(json.dumps(metadata,indent=2)+'\n')
    if (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes() or (BUILD/'nvrtc_input.cu').read_bytes()!=source.encode():
        raise RuntimeError('Observed-output v3 compiled source copies changed')
    return ptx,metadata

class ObservedOutputRuntime(ObservedAdjointRuntime):
    def __init__(self,api,device,observed_runtime,force_full=False):
        if type(force_full) is not bool:raise ValueError('force_full must be explicit bool')
        self.module=C.c_void_p();self.force_full=force_full
        self.torch,self.device=api,api.device(device)
        if self.device.type!='cuda' or self.device.index is None:raise ValueError('Explicit single CUDA device required')
        if observed_runtime.torch is not api or observed_runtime.device!=self.device:
            raise ValueError('Observed-output support must belong to the same API/device/context')
        self.support=observed_runtime.support;self.pairs=observed_runtime.pairs
        self.csr_ptr=observed_runtime.csr_ptr;self.csr_col=observed_runtime.csr_col
        self.support_metadata=observed_runtime.support_metadata
        if self.support.shape!=(self.pairs,3721) or self.support.dtype!=api.bool or self.support.device!=self.device:
            raise ValueError('Complete exact support storage required')
        if (self.csr_ptr.shape!=(self.pairs,62) or self.csr_ptr.dtype!=api.int32 or self.csr_ptr.device!=self.device
            or self.csr_col.ndim!=1 or self.csr_col.dtype!=api.int32 or self.csr_col.device!=self.device):
            raise ValueError('Complete exact packed CSR coordinates required')
        self.runtime_source_before=sha(RUNTIME_SOURCE)
        ptx,self.metadata=build_observed_output();self.path=BUILD
        self.artifacts={name:sha(BUILD/name) for name in ARTIFACTS};self.driver=cuda_driver()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p]);self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with api.cuda.device(self.device):
            api.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob));self.functions={}
            try:
                handle=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(handle),self.module,b'observed_output_product'))
                self.functions['observed_output_product']=handle
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module);self.module=C.c_void_p()
                if cleanup:raise RuntimeError('Observed-output lookup and partial cleanup failed: '+str(cleanup)) from error
                raise

    def product(self,a,b,indices):
        validate_operands(self.torch,self.device,a,b)
        if indices.shape!=(a.shape[0],) or indices.dtype!=self.torch.int64 or indices.device!=self.device or indices.stride(0)<0:
            raise ValueError('Observed-output requires original nonnegative-stride int64 slots')
        out=self.torch.empty(a.shape,dtype=self.torch.float64,device=self.device)
        arguments=[C.c_uint64(x.data_ptr()) for x in (a,b,indices,self.csr_ptr,self.csr_col,out)]
        arguments.extend(C.c_int64(s) for s in (*a.stride(),*b.stride(),indices.stride(0)))
        arguments.extend((C.c_int(a.shape[0]),C.c_int(self.pairs),C.c_int(self.force_full)))
        self._launch('observed_output_product',arguments,(a.shape[0],1,1),(256,1,1))
        return out

    def manifest(self):
        after={name:sha(BUILD/name) for name in ARTIFACTS}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256'] or sha(RUNTIME_SOURCE)!=self.runtime_source_before:
            raise RuntimeError('Observed-output source/build changed')
        counts=self.support_metadata['nonzero_pattern_counts']
        return dict(self.metadata,schema='kaksforge.observed_output_product_round6/3',build=str(BUILD),
            prepared_support=self.support_metadata,artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,
            artifact_unchanged=True,forced_full_fallback=self.force_full,full_dense_storage=True,
            unobserved_cells_zero=not self.force_full,unobserved_zero_scope='Valid slots and nonforced selected mode only',
            observed_entries_original_full64_FMA=True,all_input_states_preserved=True,
            support_buffer_reused=True,packed_CSR_output_coordinates_reused=True,
            support_extra_host_upload=False,support_upload_payload_SHA256=self.support_metadata['SHA256'],
            invalid_slots_full_product_fallback=True,no_finite_certificate_needed=True,
            actual_output_scope='Full output control' if self.force_full else 'Observed output cells for valid slots; invalid slots full output',
            scope='Budget0 final forward product only; no later squaring/norm/full-output consumer permitted',
            mean_selected_output_cells=sum(counts)/len(counts),total_selected_output_cells=sum(counts),
            original_full_output_cells_per_pair=3721,actual_bytewise_public_parity_requires_GPU_gate=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.build:
        ptx,metadata=build_observed_output(allow_compile=True)
        print(json.dumps(dict(PASS=True,build=str(BUILD),ptx_bytes=len(ptx),ptx_sha256=metadata['ptx_sha256'],CUDA_context_created=False)))
    else:print(json.dumps(dict(source=str(SOURCE),GPU_initialized=False,cached_build_exists=(BUILD/'build.json').is_file())))

if __name__=='__main__':main()
