"""Independent all-FP64 batched GEMM candidate, current Torch stream/capture.

Input/output/accumulator are FP64. Explicit FMA changes rounding/summation
relative to cuBLAS, requiring primitive, original HP28 and full-fit gates.
No GPU context or compilation is created until GemmRuntime is constructed.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
from pathlib import Path
from cuda_driver import compile_ptx

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/gemm_fp64_v1.cu'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class GemmRuntime:
    def __init__(self,torch,device,tile=32):
        if tile not in (16,32): raise ValueError('tile must be 16 or 32')
        self.torch,self.device,self.tile=torch,torch.device(device),tile
        if self.device.type!='cuda': raise ValueError('CUDA required')
        self.path=ROOT/'build/gpu-gemm-round2-fp64-v1'
        self.path.mkdir(parents=True,exist_ok=True)
        src=SOURCE.read_text(encoding='utf-8');source_sha=sha(SOURCE)
        compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
        self.metadata_path=self.path/'build.json';ptx_path=self.path/'gemm.ptx'
        if self.metadata_path.exists():
            self.metadata=json.loads(self.metadata_path.read_text())
            if (self.metadata['source_sha256']!=source_sha or self.metadata['options']!=OPTIONS
                or self.metadata['nvrtc_sha256']!=historical_compiler_sha(compiler)):
                raise RuntimeError('Existing GEMM build identity changed; preserve the old build')
            ptx=ptx_path.read_bytes()
            if hashlib.sha256(ptx).hexdigest()!=self.metadata['ptx_sha256']:
                raise RuntimeError('GEMM PTX identity changed')
        else:
            ptx,log=compile_ptx(src,OPTIONS)
            ptx_path.write_bytes(ptx);(self.path/'gemm.cu').write_bytes(SOURCE.read_bytes())
            (self.path/'build.log').write_text(log,encoding='utf-8')
            self.metadata={'source_sha256':source_sha,'ptx_sha256':hashlib.sha256(ptx).hexdigest(),
                'options':OPTIONS,'nvrtc_sha256':historical_compiler_sha(compiler),'precision':'FP64 input/output/accumulation',
                'arithmetic':'Explicit double __fma_rn, k=0..63 zero-padded serial accumulation',
                'CUDA_context_created_by_compilation':False,'fast_math':False}
            self.metadata_path.write_text(json.dumps(self.metadata,indent=2)+'\n',encoding='utf-8')
        if (self.path/'gemm.cu').read_bytes()!=SOURCE.read_bytes():
            raise RuntimeError('GEMM recorded source bytes changed')
        self.artifacts={name:sha(self.path/name) for name in ('build.json','gemm.cu','gemm.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        self._bind('cuModuleUnload',[C.c_void_p])
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device)
            blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            self.functions={}
            for name in ('gemm_tile16','gemm_tile32'):
                f=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(f),self.module,name.encode()))
                self.functions[name]=f
        runtime=self
        class BatchedMatmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx,a,b):
                ctx.save_for_backward(a,b)
                return runtime.product(a,b)
            @staticmethod
            def backward(ctx,g):
                a,b=ctx.saved_tensors
                ga=runtime.product(g,b.transpose(-1,-2)) if ctx.needs_input_grad[0] else None
                gb=runtime.product(a.transpose(-1,-2),g) if ctx.needs_input_grad[1] else None
                return ga,gb
        self.matmul=BatchedMatmul.apply

    def _bind(self,name,args):
        f=getattr(self.driver,name);f.argtypes=args;f.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('CUDA driver operation failed: '+str(code))
    def product(self,a,b):
        t=self.torch
        if (a.ndim!=3 or b.shape!=a.shape or a.shape[-2:]!=(61,61)
            or a.dtype!=t.float64 or b.dtype!=t.float64 or a.device!=b.device
            or a.device!=self.device or any(s<0 for s in (*a.stride(),*b.stride()))):
            raise ValueError('GEMM requires equally shaped (batch,61,61) FP64 CUDA tensors')
        out=t.empty(a.shape,dtype=t.float64,device=a.device)
        arguments=[C.c_uint64(x.data_ptr()) for x in (a,b,out)]
        arguments.extend(C.c_int64(s) for s in (*a.stride(),*b.stride()))
        arguments.append(C.c_int(a.shape[0]))
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        stream=t.cuda.current_stream(self.device).cuda_stream;grid=(61+self.tile-1)//self.tile
        self._check(self.driver.cuLaunchKernel(self.functions['gemm_tile'+str(self.tile)],
            grid,grid,a.shape[0],16,16,1,0,stream,pointers,None))
        return out
    def manifest(self):
        after={name:sha(self.path/name) for name in self.artifacts}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256']:
            raise RuntimeError('GEMM source/build changed during execution')
        return dict(self.metadata,tile=self.tile,build=str(self.path),backward='dA=g B^T, dB=A^T g; same FP64 kernel',
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            precision_change=False,cuBLAS_bit_parity_claimed=False)
