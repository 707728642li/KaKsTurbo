"""New shared-D fused-certificate layout; immutable observed v1 untouched."""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import argparse
from array import array
import ctypes as C
import hashlib
import json
import time
from pathlib import Path

from cuda_driver import compile_ptx
from gpu_observed_adjoint_round6 import (ObservedAdjointRuntime, observed_support,
    build_observed_adjoint, OPTIONS, ORIGINAL_SOURCE, RUNTIME_SOURCE as ORIGINAL_RUNTIME_SOURCE,
    BUILD as ORIGINAL_BUILD, ARTIFACTS as ORIGINAL_ARTIFACTS, SOURCE as ORIGINAL_OBSERVED_SOURCE)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round6/observed_adjoint_shared_v2.cu'
RUNTIME_SOURCE=Path(__file__).resolve()
BUILD=ROOT/'build/gpu-observed-adjoint-round6-shared-fp64-v2'
ARTIFACTS=('build.json','source_template.cu','nvrtc_input.cu','observed_adjoint_shared.ptx')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build_observed_shared_adjoint(allow_compile=False):
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    source=SOURCE.read_text(encoding='utf-8')
    identity={'source_sha256':sha(SOURCE),'NVRTC_input_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'runtime_source_sha256':sha(RUNTIME_SOURCE),'original_observed_runtime_sha256':sha(ORIGINAL_RUNTIME_SOURCE),
        'original_source_sha256':sha(ORIGINAL_SOURCE),'options':OPTIONS,
        'nvrtc_sha256':historical_compiler_sha(compiler),'nvrtc_builtins_sha256':historical_compiler_sha(builtins)}
    BUILD.mkdir(parents=True,exist_ok=True);record=BUILD/'build.json';ptx_path=BUILD/'observed_adjoint_shared.ptx'
    if record.exists():
        metadata=json.loads(record.read_text())
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve shared observed-adjoint build; identity changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Shared observed-adjoint PTX changed')
    else:
        if not allow_compile:raise RuntimeError('Cached shared observed-adjoint PTX required; use CPU-only --build')
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete shared observed-adjoint build')
        ptx,log=compile_ptx(source,OPTIONS)
        ptx_path.write_bytes(ptx);(BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'nvrtc_input.cu').write_bytes(source.encode());(BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            schema='kaksforge.observed_first_adjoint_shared_build_round6/2',
            precision='FP64 inputs/output/accumulator; no fast-math/TF32',
            arithmetic='Same ascending retained k FMA and positivezero tail; unsafe original k0..63',
            safety='Full finite/support certificate fused into dense shared load for every product row',
            layout='One256-thread CTA per request, shared FP64[61,65] dense operand; logical output remains61x61',
            CUDA_context_created_by_compilation=False,optimizer_or_MA_changed=False,precision_change=False)
        record.write_text(json.dumps(metadata,indent=2)+'\n')
    if (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes() or (BUILD/'nvrtc_input.cu').read_bytes()!=source.encode():
        raise RuntimeError('Shared observed-adjoint source copies changed')
    return ptx,metadata

class ObservedSharedAdjointRuntime(ObservedAdjointRuntime):
    def __init__(self,api,device,prepared,force_full=False,layout='shared'):
        if type(force_full) is not bool:raise ValueError('force_full must be explicit bool')
        if layout not in ('shared','original'):raise ValueError('observed layout must be shared or original')
        self.layout=layout;self.runtime_source_before=sha(RUNTIME_SOURCE)
        self.original_runtime_source_before=sha(ORIGINAL_RUNTIME_SOURCE)
        self.module=C.c_void_p();self.force_full=force_full
        self.torch,self.device=api,api.device(device)
        if self.device.type!='cuda' or self.device.index is None:raise ValueError('Explicit single CUDA device required')
        started=time.perf_counter();host=observed_support(prepared)
        self.host_support_prepare_seconds=time.perf_counter()-started
        self.support_metadata={'SHA256':host['SHA256'],'pairs':host['pairs'],'nonzero_pattern_counts':host['counts'],
            'slot_mapping':'pair=slot//14; all14models retained','support':'Exact prepared z0,z1 with weight>=1e-6; no state cutoff'}
        self.pairs=host['pairs']
        # Avoid NativeCUDAArrayAPI.tensor's recursive scalar flatten on3.7M
        # Python bool objects. Preserve the original complete binary support
        # identity and upload only contiguous typed host buffers.
        started=time.perf_counter()
        csr_ptr=array('i',[v for row in host['csr_ptr'] for v in row]);csr_col=array('i',host['csr_col'])
        csc_ptr=array('i',[v for row in host['csc_ptr'] for v in row]);csc_row=array('i',host['csc_row'])
        mask=b''.join(bytes(row) for row in host['mask'])
        encoded=csr_ptr.tobytes()+csr_col.tobytes()+csc_ptr.tobytes()+csc_row.tobytes()+mask
        if hashlib.sha256(encoded).hexdigest()!=host['SHA256']:
            raise RuntimeError('Packed support upload differs from exact original observed identity')
        self.host_support_pack_seconds=time.perf_counter()-started
        started=time.perf_counter()
        def upload(buffer,shape,dtype):
            return api.frombuffer(buffer,dtype=dtype).reshape(shape).to(self.device)
        self.csr_ptr=upload(csr_ptr,(self.pairs,62),api.int32)
        self.csr_col=upload(csr_col,(len(csr_col),),api.int32)
        self.csc_ptr=upload(csc_ptr,(self.pairs,62),api.int32)
        self.csc_row=upload(csc_row,(len(csc_row),),api.int32)
        self.support=upload(mask,(self.pairs,3721),api.bool)
        self.device_support_upload_seconds=time.perf_counter()-started
        if self.layout=='shared':
            ptx,self.metadata=build_observed_shared_adjoint();self.path=BUILD;self.artifact_names=ARTIFACTS
            functions=('observed_adjoint_safe','observed_adjoint_left_shared','observed_adjoint_right_shared')
        else:
            ptx,self.metadata=build_observed_adjoint();self.path=ORIGINAL_BUILD;self.artifact_names=ORIGINAL_ARTIFACTS
            functions=('observed_adjoint_safe','observed_adjoint_left','observed_adjoint_right')
        self.artifacts={name:sha(self.path/name) for name in self.artifact_names};self.driver=cuda_driver()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p]);self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with api.cuda.device(self.device):
            api.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob));self.functions={}
            try:
                for name in functions:
                    handle=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(handle),self.module,name.encode()))
                    self.functions[name]=handle
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module);self.module=C.c_void_p()
                if cleanup:raise RuntimeError('Shared observed-adjoint lookup and partial cleanup failed: '+str(cleanup)) from error
                raise

    def _product(self,g,dense,indices,right):
        if self.layout=='original':
            return super()._product(g,dense,indices,right)
        self._validate(g,dense,indices)
        out=self.torch.empty(g.shape,dtype=self.torch.float64,device=self.device)
        safe=self.torch.empty((g.shape[0],),dtype=self.torch.int32,device=self.device)
        ptr,col=(self.csc_ptr,self.csc_row) if right else (self.csr_ptr,self.csr_col)
        arguments=[C.c_uint64(x.data_ptr()) for x in (g,dense,indices,ptr,col,self.support,safe,out)]
        arguments.extend(C.c_int64(s) for s in (*g.stride(),*dense.stride(),indices.stride(0)))
        arguments.extend((C.c_int(g.shape[0]),C.c_int(self.pairs),C.c_int(self.force_full)))
        self._launch('observed_adjoint_right_shared' if right else 'observed_adjoint_left_shared',arguments,
                     (g.shape[0],1,1),(256,1,1))
        self.last_row_safety=safe
        return out

    def manifest(self):
        after={name:sha(self.path/name) for name in self.artifact_names}
        module_source=SOURCE if self.layout=='shared' else ORIGINAL_OBSERVED_SOURCE
        if (after!=self.artifacts or sha(module_source)!=self.metadata['source_sha256']
            or sha(RUNTIME_SOURCE)!=self.runtime_source_before
            or sha(ORIGINAL_RUNTIME_SOURCE)!=self.original_runtime_source_before):
            raise RuntimeError('Shared observed-adjoint source/build changed')
        return dict(self.metadata,schema='kaksforge.observed_first_adjoint_packed_round6/2',build=str(self.path),
            packed_runtime_source_sha256=self.runtime_source_before,
            original_support_runtime_sha256=self.original_runtime_source_before,
            prepared_support=self.support_metadata,artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,
            artifact_unchanged=True,forced_full_fallback=self.force_full,
            observed_layout=self.layout,row_safety_checked_every_product=True,certificate_fused_with_product=self.layout=='shared',
            standalone_safety_API_inherited=True,full_dense_outputs=True,all14_models_preserved=True,
            support_upload_path='Packed contiguous int32 arrays/bool bytes via NativeCUDAArrayAPI.frombuffer; no recursive tensor flatten',
            support_upload_payload_SHA256=self.support_metadata['SHA256'],
            host_support_prepare_seconds=self.host_support_prepare_seconds,
            host_support_pack_seconds=self.host_support_pack_seconds,
            device_support_upload_seconds=self.device_support_upload_seconds,
            actual_bytewise_parity_requires_GPU_gate=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.build:
        ptx,metadata=build_observed_shared_adjoint(allow_compile=True)
        print(json.dumps({'PASS':True,'build':str(BUILD),'ptx_bytes':len(ptx),'ptx_sha256':metadata['ptx_sha256'],'CUDA_context_created':False}))
    else:print(json.dumps({'source':str(SOURCE),'GPU_initialized':False,'cached_build_exists':(BUILD/'build.json').is_file()}))

if __name__=='__main__':main()
