"""Transpose-aware cooperative-load FP64 GEMM; same serial FMA products.

The logical tile and every k0..63 FMA are unchanged. Input lanes follow the
lower-stride dimension, then store to padded shared tiles before computing.
No Torch/CUDA import, context or build occurs until explicitly requested.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
from pathlib import Path
from cuda_driver import compile_ptx
from gpu_small_gemm_round2 import GemmRuntime, SOURCE as ORIGINAL_SOURCE

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/gemm_coalesced_fp64_v1.cu'
BUILD=ROOT/'build/gpu-gemm-coalesced-round2-fp64-v1'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def build_coalesced_gemm():
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll';builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    source=SOURCE.read_text(encoding='utf-8')
    identity={'source_sha256':sha(SOURCE),'NVRTC_input_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'original_source_sha256':sha(ORIGINAL_SOURCE),'options':OPTIONS,'nvrtc_sha256':historical_compiler_sha(compiler),'nvrtc_builtins_sha256':historical_compiler_sha(builtins)}
    BUILD.mkdir(parents=True,exist_ok=True);metadata_path=BUILD/'build.json';ptx_path=BUILD/'gemm_coalesced.ptx'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve coalesced GEMM build; source/compiler/options changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Coalesced GEMM PTX changed')
    else:
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete coalesced GEMM build')
        ptx,log=compile_ptx(source,OPTIONS);ptx_path.write_bytes(ptx)
        (BUILD/'gemm_coalesced.cu').write_bytes(SOURCE.read_bytes());(BUILD/'nvrtc_input.cu').write_bytes(source.encode())
        (BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),precision='All FP64 input/output/accumulation; no fast-math/TF32',
            arithmetic='Original explicit __fma_rn and k0..63 serial accumulation/zero padding unchanged',
            load_layout='8 specialized16/32 NN/NT/TN/TT entrypoints; each lower-stride dimension coalesced',
            shared_padding={'A_K_stride':17,'B_column_stride':'TILE+1'},CUDA_context_created_by_compilation=False,
            CUDA_module_lifetime='process-owned',optimizer_or_MA_changed=False,precision_change=False)
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'gemm_coalesced.cu').read_bytes()!=SOURCE.read_bytes() or (BUILD/'nvrtc_input.cu').read_bytes()!=source.encode():
        raise RuntimeError('Coalesced GEMM source copies changed')
    return ptx,metadata


class CoalescedGemmRuntime(GemmRuntime):
    def __init__(self,torch,device,tile=32):
        if tile not in (16,32):raise ValueError('tile must be16 or32')
        self.torch,self.device,self.tile=torch,torch.device(device),tile
        if self.device.type!='cuda':raise ValueError('CUDA required')
        ptx,self.metadata=build_coalesced_gemm();self.path=BUILD
        self.artifacts={name:sha(BUILD/name) for name in ('build.json','gemm_coalesced.cu','nvrtc_input.cu','gemm_coalesced.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p]);self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob));self.functions={}
            try:
                for size in (16,32):
                    for suffix in ('nn','nt','tn','tt'):
                        name='gemm_tile'+str(size)+'_'+suffix;function=C.c_void_p()
                        self._check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()));self.functions[name]=function
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module)
                if cleanup:raise RuntimeError('Coalesced GEMM lookup and partial module cleanup failed: '+str(cleanup)) from error
                self.module=C.c_void_p();raise
        runtime=self
        class BatchedMatmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx,a,b):ctx.save_for_backward(a,b);return runtime.product(a,b)
            @staticmethod
            def backward(ctx,g):
                a,b=ctx.saved_tensors
                ga=runtime.product(g,b.transpose(-1,-2)) if ctx.needs_input_grad[0] else None
                gb=runtime.product(a.transpose(-1,-2),g) if ctx.needs_input_grad[1] else None
                return ga,gb
        self.matmul=BatchedMatmul.apply
    def product(self,a,b):
        t=self.torch
        if (a.ndim!=3 or b.shape!=a.shape or a.shape[-2:]!=(61,61)
            or a.dtype!=t.float64 or b.dtype!=t.float64 or a.device!=b.device
            or a.device!=self.device or any(s<0 for s in (*a.stride(),*b.stride()))):
            raise ValueError('GEMM requires equally shaped(batch,61,61) FP64 CUDA tensors')
        out=t.empty(a.shape,dtype=t.float64,device=a.device)
        arguments=[C.c_uint64(x.data_ptr()) for x in (a,b,out)]
        arguments.extend(C.c_int64(s) for s in (*a.stride(),*b.stride()));arguments.append(C.c_int(a.shape[0]))
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        suffix=('t' if a.stride(-2)<a.stride(-1) else 'n')+('t' if b.stride(-2)<b.stride(-1) else 'n')
        stream=t.cuda.current_stream(self.device).cuda_stream;grid=(61+self.tile-1)//self.tile
        self._check(self.driver.cuLaunchKernel(self.functions['gemm_tile'+str(self.tile)+'_'+suffix],
            grid,grid,a.shape[0],16,16,1,0,stream,pointers,None))
        return out
    def manifest(self):
        after={name:sha(BUILD/name) for name in self.artifacts}
        if (after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256']
            or sha(ORIGINAL_SOURCE)!=self.metadata['original_source_sha256']):raise RuntimeError('Coalesced GEMM artifacts changed')
        return dict(self.metadata,tile=self.tile,build=str(BUILD),backward='Full dA=g B^T, dB=A^T g; same FP64 kernel, no gradient mask',
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            original_GEMM_bitwise_order_retained=True,cuBLAS_bit_parity_claimed=False)
