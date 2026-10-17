"""Isolated exact integer power-of-two scaling for the full native FP64 T18.

The already gated complete native orchestration is inherited unchanged. Only
the valid0..16 selected scale becomes an exact FP64 exponent-bit construction;
required/norm/correction/failure decisions and all products remain unchanged.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
from cuda_driver import compile_ptx
from gpu_native_t18_round2 import NativeT18Helpers, NativeFusedT18Runtime, SOURCE as ORIGINAL_SOURCE
from gpu_forward_matrix_fusion_round2 import ForwardMatrixFusionRuntime
from gpu_small_gemm_round2 import GemmRuntime
from gpu_sparse_t18_round2 import SparseRuntime
from gpu_t18_fusion_round2 import FusionRuntime
from torch_graph_likelihood import THETA18_FP64

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/native_t18_intscale_helpers_v1.cu'
BUILD=ROOT/'build/gpu-native-intscale-t18-round2-fp64-v1'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def kernel_source():
    return SOURCE.read_text(encoding='utf-8').replace('LOG2_THETA_VALUE',repr(math.log2(THETA18_FP64))).replace('THETA_VALUE',repr(THETA18_FP64))
def build_native_intscale_helpers():
    source=kernel_source();compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    identity={'template_sha256':sha(SOURCE),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'original_template_sha256':sha(ORIGINAL_SOURCE),'options':OPTIONS,'nvrtc_sha256':historical_compiler_sha(compiler),
        'nvrtc_builtins_sha256':historical_compiler_sha(builtins),'theta':THETA18_FP64,'log2_theta':math.log2(THETA18_FP64)}
    BUILD.mkdir(parents=True,exist_ok=True);metadata_path=BUILD/'build.json';ptx_path=BUILD/'native_t18_intscale.ptx'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve integer scale build; source/compiler/options changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Integer scale PTX changed')
    else:
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete integer scale build')
        ptx,log=compile_ptx(source,OPTIONS);ptx_path.write_bytes(ptx)
        (BUILD/'native_t18_intscale.cu').write_bytes(source.encode());(BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            precision='All FP64 input/output/arithmetic/reductions; exact exponent-bit2^-selected; no fast-math/TF32',
            scale_precondition='selected finite integer0..16 after existing norm-valid decision; invalid selected falls back0',
            required_norm_guard_changed=False,CUDA_context_created_by_compilation=False,CUDA_module_lifetime='process-owned',
            optimizer_or_MA_changed=False,higher_order_gradients=False,
            numerical_scope='Same full native T18 equations and float addition order; selected2^-k CPU bit exact for every k0..16; actual GPU gates still required')
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'native_t18_intscale.cu').read_bytes()!=source.encode() or (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes():
        raise RuntimeError('Integer scale compiled source copy changed')
    return ptx,metadata


class NativeIntScaleT18Helpers(NativeT18Helpers):
    def __init__(self,torch,device):
        self.torch,self.device=torch,torch.device(device);ptx,self.metadata=build_native_intscale_helpers()
        self.artifacts={name:sha(BUILD/name) for name in ('build.json','native_t18_intscale.cu','source_template.cu','native_t18_intscale.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p]);self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob));self.functions={}
            try:
                for name in ('native_norm_scale','native_add2','native_add3','native_add_pair','native_square_select',
                    'native_valid_exp','native_mask_gradient','native_reverse_square','native_add3_scale','native_pack'):
                    function=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()))
                    self.functions[name]=function
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module)
                if cleanup:raise RuntimeError('Integer scale helper lookup and partial module cleanup failed: '+str(cleanup)) from error
                self.module=C.c_void_p();raise
    def manifest(self):
        after={name:sha(BUILD/name) for name in self.artifacts}
        if (after!=self.artifacts or sha(SOURCE)!=self.metadata['template_sha256']
            or sha(ORIGINAL_SOURCE)!=self.metadata['original_template_sha256']):raise RuntimeError('Integer scale helper artifacts changed')
        return dict(self.metadata,artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            build=str(BUILD),forward_bitwise_parity_claimed=False,gradient_bitwise_parity_claimed=False)


class NativeIntScaleFusedT18Runtime(NativeFusedT18Runtime):
    def __init__(self,owner,objective_runtime):
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        self.matrix=ForwardMatrixFusionRuntime(owner);self.gemm=GemmRuntime(self.torch,self.device,32)
        self.sparse=SparseRuntime(self.torch,self.device,self.gemm,owner.prepared.edges)
        self.coefficients=FusionRuntime(self.torch,self.device);self.helpers=NativeIntScaleT18Helpers(self.torch,self.device)
        self.objective=objective_runtime
    def manifest(self):
        result=super().manifest();result['schema']='kaksforge.native_full_T18_exact_intscale_round2/1'
        result['full_native_forward_gradient_method_inherited_unchanged']=True
        return result
