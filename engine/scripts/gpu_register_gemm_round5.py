"""New exact-order register-blocked 61x61 batched GEMM candidates.

No CUDA load, context, GPU execution or compilation occurs on import. The two
layouts retain every round2 k0..63 explicit FP64 FMA and zero padding; actual
GPU bits, HP28 and full fits remain required before scientific acceptance.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import inspect
import json
from pathlib import Path
from cuda_driver import compile_ptx

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round5/gemm_register_fp64_v1.cu'
BUILD=ROOT/'build/gpu-gemm-round5-register-fp64-v1'
ORIGINAL_SOURCE=ROOT/'src/gpu_round2/gemm_coalesced_fp64_v1.cu'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
LAYOUTS={'whole64':(64,64,16,16),'rect32x64':(32,64,32,8)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def build_register_gemm():
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    source=SOURCE.read_text(encoding='utf-8')
    identity={'source_sha256':sha(SOURCE),'NVRTC_input_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'original_source_sha256':sha(ORIGINAL_SOURCE),'options':OPTIONS,
        'nvrtc_sha256':historical_compiler_sha(compiler),'nvrtc_builtins_sha256':historical_compiler_sha(builtins)}
    BUILD.mkdir(parents=True,exist_ok=True);record=BUILD/'build.json';ptx_path=BUILD/'gemm_register.ptx'
    if record.exists():
        metadata=json.loads(record.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve register GEMM build; identity changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Register GEMM PTX changed')
    else:
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete register GEMM build')
        ptx,log=compile_ptx(source,OPTIONS)
        ptx_path.write_bytes(ptx);(BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'nvrtc_input.cu').write_bytes(source.encode());(BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),precision='FP64 input/output/accumulator; no fast-math/TF32',
            arithmetic='Same +0 initialization and serial k0..63 __fma_rn; no split-K or reassociation',
            CUDA_context_created_by_compilation=False,optimizer_or_MA_changed=False,precision_change=False,
            layouts={k:dict(output_tile=list(v[:2]),threads=list(v[2:]),outputs_per_thread=(v[0]//v[3])*(v[1]//v[2])) for k,v in LAYOUTS.items()})
        record.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes() or (BUILD/'nvrtc_input.cu').read_bytes()!=source.encode():
        raise RuntimeError('Register GEMM source copies changed')
    return ptx,metadata

def _span(value):
    start=value.data_ptr();end=start+8*(1+sum((n-1)*s for n,s in zip(value.shape,value.stride())))
    return start,end

def validate_operands(api,device,a,b,out=None):
    if (a.ndim!=3 or b.shape!=a.shape or a.shape[-2:]!=(61,61) or not 1<=a.shape[0]<=1048576
        or a.dtype!=api.float64 or b.dtype!=api.float64 or a.device!=b.device or a.device!=device
        or any(s<0 for s in (*a.stride(),*b.stride()))):
        raise ValueError('GEMM requires equally shaped(batch,61,61) FP64 CUDA tensors with nonnegative strides')
    if out is not None:
        if out.shape!=a.shape or out.dtype!=api.float64 or out.device!=device or not out.is_contiguous():
            raise ValueError('Output requires separate contiguous FP64 CUDA storage of the same shape')
        low,high=_span(out)
        for value in (a,b):
            begin,end=_span(value)
            if low<end and begin<high:raise ValueError('GEMM output must not overlap either input')

class RegisterGemmRuntime:
    def __init__(self,api,device,layout='rect32x64'):
        if layout not in LAYOUTS:raise ValueError('Unknown register GEMM layout')
        self.torch,self.device,self.layout=api,api.device(device),layout
        if self.device.type!='cuda':raise ValueError('CUDA required')
        ptx,self.metadata=build_register_gemm();self.path=BUILD
        self.artifacts={n:sha(BUILD/n) for n in ('build.json','source_template.cu','nvrtc_input.cu','gemm_register.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p]);self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with api.cuda.device(self.device):
            api.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob));self.functions={}
            try:
                for name in LAYOUTS:
                    for suffix in ('nn','nt','tn','tt'):
                        key='gemm_'+name+'_'+suffix;handle=C.c_void_p()
                        self._check(self.driver.cuModuleGetFunction(C.byref(handle),self.module,key.encode()));self.functions[key]=handle
            except BaseException:
                self._check(self.driver.cuModuleUnload(self.module));self.module=C.c_void_p();raise
    def _bind(self,name,args):
        fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('Register GEMM CUDA Driver operation failed: '+str(code))
    def product_into(self,a,b,out):
        validate_operands(self.torch,self.device,a,b,out)
        return self._launch(a,b,out)
    def _launch(self,a,b,out):
        arguments=[C.c_uint64(value.data_ptr()) for value in (a,b,out)]
        arguments.extend(C.c_int64(v) for v in (*a.stride(),*b.stride()));arguments.append(C.c_int(a.shape[0]))
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        suffix=('t' if a.stride(-2)<a.stride(-1) else 'n')+('t' if b.stride(-2)<b.stride(-1) else 'n')
        rows,cols,tx,ty=LAYOUTS[self.layout];stream=self.torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.functions['gemm_'+self.layout+'_'+suffix],
            (61+cols-1)//cols,(61+rows-1)//rows,a.shape[0],tx,ty,1,0,stream,pointers,None))
        return out
    def product(self,a,b):
        validate_operands(self.torch,self.device,a,b)
        return self._launch(a,b,self.torch.empty(a.shape,dtype=self.torch.float64,device=self.device))
    def manifest(self):
        after={n:sha(BUILD/n) for n in self.artifacts}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256'] or sha(ORIGINAL_SOURCE)!=self.metadata['original_source_sha256']:
            raise RuntimeError('Register GEMM source/build changed')
        return dict(self.metadata,schema='kaksforge.register_blocked_GEMM_round5/1',layout=self.layout,build=str(BUILD),
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            mathematical_tape_changed=False,original_GEMM_bitwise_order_retained=True,actual_bitwise_parity_requires_GPU_gate=True,
            beta='No beta/read-C: writes each contiguous output once; product always allocates disjoint output',
            output_alias_policy='Reject overlap; shared A/B and repeated broadcast inputs accepted')

def install_register_gemm(provider,layout='rect32x64'):
    """Replace execution layout before any tape/graph exists; no tape edits."""
    if provider.gradient_buckets or provider._pending:raise RuntimeError('GEMM installer requires a fresh provider with no tapes')
    from gpu_native_t18_round2 import NativeFusedT18Runtime
    tape=provider.native_runtime.forward_gradient.__func__
    if tape is not NativeFusedT18Runtime.forward_gradient:raise RuntimeError('Inherited mathematical tape changed')
    provider.api.synchronize(timeout=10)
    old=provider.native_runtime.gemm;new=RegisterGemmRuntime(provider.api,provider.device,layout)
    try:
        old._check(old.driver.cuModuleUnload(old.module));old.module=C.c_void_p()
    except BaseException:
        new._check(new.driver.cuModuleUnload(new.module));new.module=C.c_void_p();raise
    provider.native_runtime.gemm=new;provider.native_runtime.sparse.dense=new
    provider.round5_gemm_installation={
        'layout':layout,'installer_source_sha256':sha(Path(__file__)),
        'forward_gradient_inherited_identical_function':True,
        'forward_gradient_source_sha256':hashlib.sha256(inspect.getsource(tape).encode()).hexdigest(),
        'replacement_scope':'Dense GEMM product method and sparse dense fallback pointer only',
        'optimizer_or_MA_changed':False}
    return new
