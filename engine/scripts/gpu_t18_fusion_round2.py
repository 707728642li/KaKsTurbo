"""Experimental FP64 T18 coefficient fusion on Torch's existing CUDA stream.

No matrix product, coefficient, scaling, or convergence rule is changed.
NVRTC kernels use explicit double arithmetic with --fmad=false. Forward keeps
the original five-term left fold; backward forms its linear transpose. All
CUDA launches use the current Torch context and stream (also during capture).
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
from pathlib import Path
import threading
from cuda_driver import compile_ptx
from torch_graph_likelihood import T18_COEFFICIENTS

ROOT=Path(__file__).resolve().parents[1]
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']

def kernel_source():
    coefficients=','.join(repr(value) for row in T18_COEFFICIENTS for value in row)
    return r'''
__device__ __constant__ double COEFF[25]={COEFFICIENTS};
extern "C" __global__ void t18_fold(const double* eye,const double* z,
 const double* a2,const double* a3,const double* a6,double* out,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 double p0=eye[i%3721],p1=z[i],p2=a2[i],p3=a3[i],p4=a6[i];
 for(int r=0;r<5;r++){
  double v=0.;v=v+p0*COEFF[r*5];v=v+p1*COEFF[r*5+1];
  v=v+p2*COEFF[r*5+2];v=v+p3*COEFF[r*5+3];v=v+p4*COEFF[r*5+4];
  out[(long long)r*n+i]=v;
 }
}
extern "C" __global__ void t18_fold_backward(const double* g0,const double* g1,
 const double* g2,const double* g3,const double* g4,double* out,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 double a=g0[i],b=g1[i],c=g2[i],d=g3[i],e=g4[i];
 for(int k=1;k<5;k++){
  double v=0.;v=v+a*COEFF[k];v=v+b*COEFF[5+k];v=v+c*COEFF[10+k];
  v=v+d*COEFF[15+k];v=v+e*COEFF[20+k];out[(long long)(k-1)*n+i]=v;
 }
}
'''.replace('COEFFICIENTS',coefficients)

class FusionRuntime:
    def __init__(self,torch,device):
        self.torch,self.device=torch,device
        source=kernel_source();self.source_sha256=hashlib.sha256(source.encode()).hexdigest()
        self.path=ROOT/'build'/'gpu-t18-fusion-round2-v1'
        self.path.mkdir(parents=True,exist_ok=True)
        self.manifest_path=self.path/'build.json'
        source_path=self.path/'t18_fold.cu';ptx_path=self.path/'t18_fold.ptx'
        if self.manifest_path.exists():
            self.metadata=json.loads(self.manifest_path.read_text())
            if self.metadata['source_sha256']!=self.source_sha256 or source_path.read_text()!=source:
                raise RuntimeError('Preserve earlier fusion build; source identity changed')
            ptx=ptx_path.read_bytes()
            if hashlib.sha256(ptx).hexdigest()!=self.metadata['ptx_sha256']:
                raise RuntimeError('Fusion PTX differs from build record')
        else:
            ptx,log=compile_ptx(source,OPTIONS)
            source_path.write_text(source,encoding='utf-8');ptx_path.write_bytes(ptx)
            (self.path/'build.log').write_text(log,encoding='utf-8')
            self.metadata={'source_sha256':self.source_sha256,'ptx_sha256':hashlib.sha256(ptx).hexdigest(),
                'options':OPTIONS,'CUDA_context_created_by_compilation':False,
                'coefficients':T18_COEFFICIENTS,'arithmetic':'double; left fold; contraction disabled',
                'nvrtc_sha256':historical_compiler_sha(ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll')}
            self.manifest_path.write_text(json.dumps(self.metadata,indent=2)+'\n',encoding='utf-8')
        self.driver=cuda_driver();self.module=C.c_void_p()
        if self.metadata['options']!=OPTIONS or [list(row) for row in self.metadata['coefficients']]!=[list(row) for row in T18_COEFFICIENTS]:
            raise RuntimeError('Fusion build options or coefficient identity mismatch')
        actual_nvrtc=historical_compiler_sha(ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll')
        if self.metadata['nvrtc_sha256']!=actual_nvrtc:raise RuntimeError('Fusion NVRTC compiler differs from recorded build')
        self.artifact_sha256={name:hashlib.sha256((self.path/name).read_bytes()).hexdigest()
            for name in ('build.json','t18_fold.cu','t18_fold.ptx')}
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        self._bind('cuModuleUnload',[C.c_void_p])
        with torch.cuda.device(device):
            torch.cuda.current_stream(device) # Initialize current primary context.
            blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            self.functions={}
            for name in ('t18_fold','t18_fold_backward'):
                fn=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
                self.functions[name]=fn
        runtime=self
        class Fold(torch.autograd.Function):
            @staticmethod
            def forward(ctx,eye,z,a2,a3,a6):
                ctx.shape=z.shape
                output=torch.empty((5,*z.shape),dtype=torch.float64,device=z.device)
                runtime.launch('t18_fold',[eye,z,a2,a3,a6,output],z.numel())
                return tuple(output.unbind(0))
            @staticmethod
            def backward(ctx,*gradients):
                gs=[g.contiguous() for g in gradients]
                output=torch.empty((4,*ctx.shape),dtype=torch.float64,device=gs[0].device)
                runtime.launch('t18_fold_backward',[*gs,output],gs[0].numel())
                return (None,*output.unbind(0))
        self.fold=Fold.apply

    def _bind(self,name,args):
        function=getattr(self.driver,name);function.argtypes=args;function.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('CUDA driver operation failed: '+str(code))
    def launch(self,name,tensors,count):
        arguments=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_int64(count)]
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(a),C.c_void_p) for a in arguments))
        stream=self.torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.functions[name],(count+255)//256,1,1,256,1,1,0,stream,pointers,None))
    def manifest(self):
        after={name:hashlib.sha256((self.path/name).read_bytes()).hexdigest() for name in self.artifact_sha256}
        if self.artifact_sha256!=after:raise RuntimeError('Fusion build artifact changed during execution')
        return dict(self.metadata,build=str(self.path),forward='five-term left fold',
            backward='linear transpose; independently validated tolerance required',fast_math=False,
            artifact_SHA256_before=self.artifact_sha256,artifact_SHA256_after=after,artifact_unchanged=True)

