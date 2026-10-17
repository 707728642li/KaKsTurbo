"""Native FP64 objective/full gExp for the independent complete native T18.

No Torch autograd. One kernel follows the original elementwise likelihood and
stepwise reverse arithmetic. Positive patterns must be unique; every count is
preserved in a dense perpair weight map. Nonobserved gradients are explicitly0.
CUDA is initialized only by NativeObjectiveRuntime; CPU NVRTC build is separate.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha

from array import array
import ctypes as C
import hashlib
import json
import math
import operator
from pathlib import Path
import sys

from cuda_driver import compile_ptx

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/native_objective_v1.cu'
RUNTIME_SOURCE=Path(__file__).resolve()
BUILD=ROOT/'build/gpu-native-objective-round2-fp64-v1'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dense_pattern_weights(prepared):
    """Pack exact original unique positive patterns, without evaluating F/Q."""
    pairs=len(prepared.ids)
    if not 1<=pairs<=100000 or any(len(getattr(prepared,name))!=pairs for name in ('z0','z1','weights')):
        raise ValueError('Expected1..100000 complete prepared pair pattern tables')
    if sys.byteorder!='little' or array('d').itemsize!=8:
        raise RuntimeError('Expected native Windows little-endian IEEE FP64 packing')
    packed=array('d',[0.])*(pairs*3721)
    pattern_count=ignored_count=0;ignored_sum=0.;total_weight=0.
    for pair,(z0,z1,weights) in enumerate(zip(prepared.z0,prepared.z1,prepared.weights)):
        if len(z0)!=len(z1) or len(z0)!=len(weights):raise ValueError('Prepared pattern dimensions disagree')
        seen=set()
        for first,second,original in zip(z0,z1,weights):
            i,j=operator.index(first),operator.index(second);weight=float(original)
            if not 0<=i<61 or not 0<=j<61 or not math.isfinite(weight) or weight<0.:
                raise ValueError('Invalid native objective pattern index/weight')
            total_weight+=weight
            if weight<1e-6:
                ignored_count+=1;ignored_sum+=weight
                continue
            cell=i*61+j
            if cell in seen:
                raise ValueError('Duplicate positive pattern requires separately gated aggregation; no count silently dropped')
            seen.add(cell);packed[pair*3721+cell]=weight;pattern_count+=1
    identity={'pairs':pairs,'entries':pairs*3721,'bytes':len(packed)*8,'positive_pattern_count':pattern_count,
        'positive_patterns_unique_per_pair':True,'duplicate_policy':'Explicit rejection; no atomics/aggregation/count loss',
        'below1e_minus6_pattern_count':ignored_count,'below1e_minus6_weight_sum':ignored_sum,
        'all_original_pattern_weight_sum':total_weight,
        'dense_weight_SHA256':hashlib.sha256(memoryview(packed).cast('B')).hexdigest()}
    return packed,identity


def build_native_objective_ptx():
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    identity={'source_sha256':sha(SOURCE),'options':OPTIONS,'nvrtc_sha256':historical_compiler_sha(compiler),'nvrtc_builtins_sha256':historical_compiler_sha(builtins)}
    BUILD.mkdir(parents=True,exist_ok=True);metadata_path=BUILD/'build.json';ptx_path=BUILD/'native_objective.ptx'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(key)!=value for key,value in identity.items()):raise RuntimeError('Preserve prior native objective build: source/compiler/options changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Native objective PTX changed')
    else:
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete prior native objective build; choose new version')
        try:ptx,log=compile_ptx(SOURCE.read_text(encoding='utf-8'),OPTIONS)
        except BaseException as error:
            (BUILD/'compile_failure.json').write_text(json.dumps(dict(identity,error_type=type(error).__name__,error=str(error),CUDA_context_created=False),indent=2)+'\n',encoding='utf-8')
            raise
        ptx_path.write_bytes(ptx);(BUILD/'native_objective.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'build.log').write_text(log,encoding='utf-8')
        lib=C.CDLL(str(compiler));version=lib.nvrtcVersion
        version.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int)];version.restype=C.c_int
        major,minor=C.c_int(),C.c_int();code=version(C.byref(major),C.byref(minor))
        if code:raise RuntimeError('nvrtcVersion failed: '+str(code))
        if sha(SOURCE)!=identity['source_sha256'] or historical_compiler_sha(compiler)!=identity['nvrtc_sha256'] or historical_compiler_sha(builtins)!=identity['nvrtc_builtins_sha256']:
            raise RuntimeError('Native objective source/compiler changed during compilation')
        metadata=dict(identity,schema='kaksforge.native_objective_FP64_build_round2/1',
            nvrtc_version=[major.value,minor.value],nvrtc_path=compiler.relative_to(ROOT).as_posix(),
            nvrtc_builtins_path=builtins.relative_to(ROOT).as_posix(),
            ptx_sha256=hashlib.sha256(ptx).hexdigest(),precision='FP64 likelihood and full exponential adjoint',
            arithmetic='Original divide/multiply/stationary/log/weight order; stepwise chain reverse',
            F_reduction='Perlane ascending densecell p+=256 leftfold;256 fixed binarytree128..1; no bitparity claim',
            gradient_reduction='Unique positive prepared pattern percell; no atomics or count aggregation',
            CUDA_context_created_by_compilation=False,fast_math=False,probability_clamp=False)
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'native_objective.cu').read_bytes()!=SOURCE.read_bytes():raise RuntimeError('Native objective compiled sourcecopy changed')
    return ptx,metadata


class NativeObjectiveRuntime:
    def __init__(self,owner):
        self.owner,self.torch,self.device=owner,owner.torch,owner.torch.device(owner.device)
        self.runtime_source_sha256=sha(RUNTIME_SOURCE)
        if self.device.type!='cuda' or self.device.index is None:raise ValueError('Explicit selected CUDA device required')
        packed,self.pattern_identity=dense_pattern_weights(owner.prepared)
        free,_=self.torch.cuda.mem_get_info(self.device)
        if len(packed)*8+64*1024**2+2*1024**3>free:raise MemoryError('Native objective resident weights plus64MiB workspace and2GiB reserve exceed free memory')
        ptx,self.metadata=build_native_objective_ptx();self.path=BUILD
        self.artifacts={name:sha(BUILD/name) for name in ('build.json','native_objective.cu','native_objective.ptx')}
        self._weights_host=packed
        self._weight_matrix=self.torch.frombuffer(packed,dtype=self.torch.float64).reshape(len(owner.prepared.ids),61,61).to(self.device)
        self.driver=cuda_driver();self.module=C.c_void_p();self.function=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        self._bind('cuModuleUnload',[C.c_void_p])
        with self.torch.cuda.device(self.device):
            self.torch.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            try:self._check(self.driver.cuModuleGetFunction(C.byref(self.function),self.module,b'native_objective'))
            except BaseException as error:
                code=self.driver.cuModuleUnload(self.module)
                if code:raise RuntimeError('Native objective functionlookup failed; moduleunload also failed: '+str(code)) from error
                self.module=C.c_void_p();raise
        if self.torch.cuda.mem_get_info(self.device)[0]<2*1024**3:
            raise MemoryError('Native objective initialization violated2GiB reserve')

    def _bind(self,name,args):
        function=getattr(self.driver,name);function.argtypes=args;function.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('CUDA native objective operation failed: '+str(code))

    def evaluate(self,indices,exponential,pi,sqrt_pi,live_outer,kind):
        """Return(raw F[rows],full dense gExp[rows,61,61] orNone forkind0)."""
        t=self.torch
        if exponential.ndim!=3:raise ValueError('Native objective exponential must have3 dimensions')
        rows=exponential.shape[0]
        if kind not in (0,1):raise ValueError('Native objective kind must be0 or1')
        if (exponential.ndim!=3 or exponential.shape!=(rows,61,61) or not 1<=rows<=65535
            or indices.shape!=(rows,) or indices.dtype!=t.int64 or pi.shape!=(rows,61) or sqrt_pi.shape!=(rows,61)
            or live_outer.shape!=(rows,61,61) or live_outer.dtype!=t.bool):raise ValueError('Invalid native objective batch/indices/pi/live dimensions')
        if any(value.dtype!=t.float64 for value in (exponential,pi,sqrt_pi)):
            raise ValueError('Native objective floating input must beFP64')
        if any(value.device!=self.device or any(stride<0 for stride in value.stride()) for value in (indices,exponential,pi,sqrt_pi,live_outer)):
            raise ValueError('Native objective inputs must use selected CUDA device and nonnegative strides')
        F=t.empty(rows,dtype=t.float64,device=self.device)
        gexp=t.empty((rows,61,61),dtype=t.float64,device=self.device) if kind else None
        arguments=[C.c_uint64(value.data_ptr()) for value in (exponential,pi,sqrt_pi,live_outer,indices,self._weight_matrix,F)]
        arguments.append(C.c_uint64(gexp.data_ptr() if gexp is not None else 0))
        for value in (exponential,pi,sqrt_pi,live_outer,indices):arguments.extend(C.c_int64(stride) for stride in value.stride())
        arguments.extend(C.c_int(value) for value in (rows,self.pattern_identity['pairs'],kind))
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(value),C.c_void_p) for value in arguments))
        with t.cuda.device(self.device):
            stream=t.cuda.current_stream(self.device).cuda_stream
            self._check(self.driver.cuLaunchKernel(self.function,rows,1,1,256,1,1,0,stream,pointers,None))
        return F,gexp

    def manifest(self):
        after={name:sha(self.path/name) for name in self.artifacts}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256'] or sha(RUNTIME_SOURCE)!=self.runtime_source_sha256:
            raise RuntimeError('Native objective source/build changed during run')
        if hashlib.sha256(memoryview(self._weights_host).cast('B')).hexdigest()!=self.pattern_identity['dense_weight_SHA256']:
            raise RuntimeError('Native objective preparedweight packing changed')
        return dict(self.metadata,build=str(self.path),runtime_source_sha256=self.runtime_source_sha256,
            runtime_source_unchanged=True,prepared_weights=self.pattern_identity,
            API='evaluate(indices,exp,pi,sqrt_pi,live_outer,kind)->(F,gExp orNone)',
            gExp='Full dense(batch,61,61),zerosforunobserved/inactive/livefalse; no factor dropped',
            absent_support='Original identity fallback before stationary multiply; finite failure left to inherited valid/error guard',
            kernel_calls_per_evaluation=1,gradient_precision='FP64 stepwise -w/fh,*pi_i,*sqrt_j,/sqrt_i',
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            CUDA_module_lifetime='Provider/graphs until isolated process exit; no unsafe destructor',
            higher_order_gradients=False,gradient_bit_parity_claimed=False,
            requires_CPU_scalar_original_HP28_and_complete_development_fit_gates=True)
