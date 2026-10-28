"""Full first-order FP64 T18 without Torch autograd/elementwise operations.

Torch owns allocations, views and CUDA stream/capture only. Complete matrix
products, all allocated squarings and unmasked adjoints remain. Root must run
the unchanged HP28 and every same1000 returned-point gate before integration.
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
from gpu_forward_matrix_fusion_round2 import ForwardMatrixFusionRuntime
from gpu_small_gemm_round2 import GemmRuntime
from gpu_sparse_t18_round2 import SparseRuntime
from gpu_t18_fusion_round2 import FusionRuntime
from torch_graph_likelihood import THETA18_FP64
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/native_t18_helpers_v1.cu'
BUILD=ROOT/'build/gpu-native-t18-round2-fp64-v1'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def kernel_source():
    return SOURCE.read_text(encoding='utf-8').replace('LOG2_THETA_VALUE',repr(math.log2(THETA18_FP64))).replace('THETA_VALUE',repr(THETA18_FP64))
def build_native_helpers():
    source=kernel_source();compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    identity={'template_sha256':sha(SOURCE),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'options':OPTIONS,'nvrtc_sha256':historical_compiler_sha(compiler),'nvrtc_builtins_sha256':historical_compiler_sha(builtins),
        'theta':THETA18_FP64,'log2_theta':math.log2(THETA18_FP64)}
    BUILD.mkdir(parents=True,exist_ok=True);metadata_path=BUILD/'build.json';ptx_path=BUILD/'native_t18.ptx'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve native T18 build; source/compiler/options changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Native T18 PTX changed')
    else:
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete native T18 build')
        ptx,log=compile_ptx(source,OPTIONS);ptx_path.write_bytes(ptx)
        (BUILD/'native_t18.cu').write_bytes(source.encode());(BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            precision='All FP64 input/output/arithmetic/reductions; no fast-math/TF32',
            CUDA_context_created_by_compilation=False,CUDA_module_lifetime='process-owned',
            optimizer_or_MA_changed=False,higher_order_gradients=False,
            numerical_scope='Complete T18 forward/reverse; native norm column sums ascending rows; different float summation order requires original gates')
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'native_t18.cu').read_bytes()!=source.encode() or (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes():
        raise RuntimeError('Native T18 compiled source copy changed')
    return ptx,metadata

class NativeT18Helpers:
    def __init__(self,torch,device):
        self.torch,self.device=torch,torch.device(device);ptx,self.metadata=build_native_helpers()
        self.artifacts={name:sha(BUILD/name) for name in ('build.json','native_t18.cu','source_template.cu','native_t18.ptx')}
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
                if cleanup:raise RuntimeError('Native helper lookup and partial module cleanup failed: '+str(cleanup)) from error
                self.module=C.c_void_p();raise
    def _bind(self,name,args):
        fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('Native T18 CUDA driver operation failed: '+str(code))
    def launch(self,name,tensors,count,extra=(),row_kernel=False,int_count=False):
        arguments=[C.c_uint64(t.data_ptr()) for t in tensors]+[(C.c_int if int_count else C.c_int64)(count)]
        arguments += [C.c_int(v) for v in extra]
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        stream=self.torch.cuda.current_stream(self.device).cuda_stream;grid=count if row_kernel else (count+255)//256
        self._check(self.driver.cuLaunchKernel(self.functions[name],grid,1,1,256,1,1,0,stream,pointers,None))
    def empty_like(self,value):return self.torch.empty(value.shape,dtype=self.torch.float64,device=self.device)
    def norm_scale(self,a,budget):
        z=self.empty_like(a);diagnostic=self.torch.empty((a.shape[0],4),dtype=self.torch.float64,device=self.device)
        self.launch('native_norm_scale',[a,z,diagnostic],a.shape[0],(budget,),row_kernel=True,int_count=True)
        return z,diagnostic
    def add(self,a,b):
        out=self.empty_like(a);self.launch('native_add2',[a,b,out],a.numel());return out
    def add3(self,a,b,c):
        out=self.empty_like(a);self.launch('native_add3',[a,b,c,out],a.numel());return out
    def add_pair(self,a,b,c,d):
        ab=self.empty_like(a);cd=self.empty_like(c)
        self.launch('native_add_pair',[a,b,c,d,ab,cd],a.numel());return ab,cd
    def square_select(self,previous,square,diagnostic,step):
        out=self.empty_like(previous);self.launch('native_square_select',[previous,square,diagnostic,out],previous.numel(),(step,));return out
    def valid_exp(self,value,diagnostic):
        out=self.empty_like(value);self.launch('native_valid_exp',[value,diagnostic,out],value.numel());return out
    def mask_gradient(self,g,diagnostic,step=-1,mode=0):
        out=self.empty_like(g);self.launch('native_mask_gradient',[g,diagnostic,out],g.numel(),(step,mode));return out
    def reverse_square(self,left,right,incoming,diagnostic,step):
        out=self.empty_like(left);self.launch('native_reverse_square',[left,right,incoming,diagnostic,out],left.numel(),(step,));return out
    def add3_scale(self,a,b,c,diagnostic):
        out=self.empty_like(a);self.launch('native_add3_scale',[a,b,c,diagnostic,out],a.numel());return out
    def pack(self,F,diagnostic,raw,x,indices,nkappa,kind):
        out=self.torch.empty((x.shape[0],11),dtype=self.torch.float64,device=self.device)
        self.launch('native_pack',[F,diagnostic,x if raw is None else raw,x,indices,nkappa,out],x.shape[0],(kind,),int_count=True)
        return out
    def manifest(self):
        after={name:sha(BUILD/name) for name in self.artifacts}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['template_sha256']:raise RuntimeError('Native helper artifacts changed during run')
        return dict(self.metadata,artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            build=str(BUILD),forward_bitwise_parity_claimed=False,gradient_bitwise_parity_claimed=False)

class NativeFusedT18Runtime:
    def __init__(self,owner,objective_runtime):
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        self.matrix=ForwardMatrixFusionRuntime(owner);self.gemm=GemmRuntime(self.torch,self.device,32)
        self.sparse=SparseRuntime(self.torch,self.device,self.gemm,owner.prepared.edges)
        self.coefficients=FusionRuntime(self.torch,self.device);self.helpers=NativeT18Helpers(self.torch,self.device)
        self.objective=objective_runtime
    def fold(self,z,a2,a3,a6):
        output=self.torch.empty((5,*z.shape),dtype=self.torch.float64,device=self.device)
        self.coefficients.launch('t18_fold',[self.owner.identity,z,a2,a3,a6,output],z.numel())
        return tuple(output.unbind(0))
    def fold_backward(self,b0,b1,b2,b3,b4):
        output=self.torch.empty((4,*b0.shape),dtype=self.torch.float64,device=self.device)
        self.coefficients.launch('t18_fold_backward',[b0,b1,b2,b3,b4,output],b0.numel())
        return tuple(output.unbind(0))
    def forward_gradient(self,indices,x,budget,kind):
        if kind not in (0,1) or not isinstance(budget,int) or not 0<=budget<=16:raise ValueError('Invalid native graph kind/budget')
        H=self.helpers;mm=self.gemm.product;left=self.sparse.product_left;right=self.sparse.product_right
        a,pi,sqrt_pi,live,s,mr,rates=self.matrix.forward(indices,x)
        z,diagnostic=H.norm_scale(a,budget)
        a2=left(z,z);a3=left(z,a2);a6=mm(a3,a3)
        b0,b1,b2,b3,b4=self.fold(z,a2,a3,a6)
        c=H.add(b3,mm(b0,b4));d=H.add(b2,c);result=H.add(b1,mm(d,c))
        tape=[result]
        for step in range(budget):
            result=H.square_select(result,mm(result,result),diagnostic,step);tape.append(result)
        exp=H.valid_exp(result,diagnostic)
        F,g_exp=self.objective.evaluate(indices,exp,pi,sqrt_pi,live,kind)
        if not kind:return H.pack(F,diagnostic,None,x,indices,self.owner.nkappa,kind)
        g=H.mask_gradient(g_exp,diagnostic)
        for step in range(budget-1,-1,-1):
            square_g=H.mask_gradient(g,diagnostic,step,1);previous_t=tape[step].transpose(-1,-2)
            g=H.reverse_square(mm(square_g,previous_t),mm(previous_t,square_g),g,diagnostic,step)
        gb2=mm(g,c.transpose(-1,-2));gb3=H.add(gb2,mm(d.transpose(-1,-2),g))
        gb0=mm(gb3,b4.transpose(-1,-2));gb4=mm(b0.transpose(-1,-2),gb3)
        gz,ga2,ga3,ga6=self.fold_backward(gb0,g,gb2,gb3,gb4)
        a3t=a3.transpose(-1,-2)
        ga3=H.add3(ga3,mm(ga6,a3t),mm(a3t,ga6))
        zt=z.transpose(-1,-2)
        gz,ga2=H.add_pair(gz,mm(ga3,a2.transpose(-1,-2)),ga2,left(zt,ga3))
        g_a=H.add3_scale(gz,right(ga2,zt),left(zt,ga2),diagnostic)
        raw=self.matrix.reverse.backward(g_a,s,a,pi,sqrt_pi,mr,rates,x,indices)
        return H.pack(F,diagnostic,raw,x,indices,self.owner.nkappa,kind)
    def manifest(self):
        return {'schema':'kaksforge.native_full_T18_round2/1','Torch_scope':'allocation/views/currentstream/CUDAGraph only; no Torch autograd or elementwise kernels',
            'forward_products':'5+budget, first2 structural sparse','reverse_products':'10+2budget, 3 structural sparse; all remaining dense',
            'symmetry_shortcut':False,'all_allocated_squarings_retained':True,'precision_change':False,
            'matrix':self.matrix.manifest(),'dense':self.gemm.manifest(),'sparse':self.sparse.manifest(),
            'coefficients':self.coefficients.manifest(),'helpers':self.helpers.manifest(),'objective':self.objective.manifest(),
            'optimizer_or_MA_changed':False,'requires_original_HP28_and_complete1000_returned_point_gates':True}
