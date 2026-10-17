"""Independent FP64 forward Q/mr/S/A fusion, existing matrix reverse kernel.

Two kernels replace dense Torch gathers/scatters/reductions/mirror/masks.
Every code1 edge and the unmasked full mean rate remain. Float operation
order per edge is retained; reduction scheduling follows local Torch headers
but bitwise identity is not assumed. HP28 and full1000 gates remain required.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
from pathlib import Path
from cuda_driver import compile_ptx
from gpu_matrix_contract_round2 import MatrixContractRuntime
from torch_matrix_prototype import codon_edges
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/matrix_forward_fusion_v1.cu'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
REDUCE_HEADER=ROOT/'envs/cuda-torch/Lib/site-packages/torch/include/ATen/native/cuda/Reduce.cuh'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def kernel_source():
    edges=codon_edges();code=[-1]*3721
    for i,j,r,syn in edges:code[i*61+j]=code[j*61+i]=r+(0 if syn else 6)
    replacements={'EDGE_CODE_VALUES':code,'EDGE_I_VALUES':[e[0] for e in edges],
        'EDGE_J_VALUES':[e[1] for e in edges],'EDGE_R_VALUES':[e[2] for e in edges],
        'EDGE_SYN_VALUES':[int(e[3]) for e in edges]}
    source=SOURCE.read_text(encoding='utf-8')
    for key,values in replacements.items():source=source.replace(key,','.join(map(str,values)))
    return source

class ForwardMatrixFusionRuntime:
    def __init__(self,owner):
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        if [tuple(v) for v in owner.prepared.edges]!=codon_edges():
            raise ValueError('Forward matrix fusion requires exact code1 edges')
        self.reverse=MatrixContractRuntime(owner)
        self.path=ROOT/'build/gpu-forward-matrix-fusion-round2-v1';self.path.mkdir(parents=True,exist_ok=True)
        metadata_path=self.path/'build.json';ptx_path=self.path/'matrix_forward.ptx'
        compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
        generated=kernel_source();generated_sha=hashlib.sha256(generated.encode()).hexdigest()
        if metadata_path.exists():
            self.metadata=json.loads(metadata_path.read_text())
            if (self.metadata['source_template_sha256']!=sha(SOURCE)
                or self.metadata['generated_source_sha256']!=generated_sha or self.metadata['options']!=OPTIONS
                or self.metadata['nvrtc_sha256']!=historical_compiler_sha(compiler) or self.metadata['torch_reduce_header_sha256']!=sha(REDUCE_HEADER)):
                raise RuntimeError('Preserve old forward fusion build; source/compiler/header changed')
            ptx=ptx_path.read_bytes()
            if hashlib.sha256(ptx).hexdigest()!=self.metadata['ptx_sha256']:
                raise RuntimeError('Forward fusion PTX changed')
        else:
            ptx,log=compile_ptx(generated,OPTIONS);ptx_path.write_bytes(ptx)
            (self.path/'matrix_forward.cu').write_text(generated,encoding='utf-8')
            (self.path/'source_template.cu').write_bytes(SOURCE.read_bytes())
            (self.path/'build.log').write_text(log,encoding='utf-8')
            self.metadata={'source_template_sha256':sha(SOURCE),'generated_source_sha256':generated_sha,
                'ptx_sha256':hashlib.sha256(ptx).hexdigest(),'nvrtc_sha256':historical_compiler_sha(compiler),'options':OPTIONS,
                'torch_reduce_header_sha256':sha(REDUCE_HEADER),'precision':'All FP64; no fast-math or TF32',
                'edge_count':263,'CUDA_context_created_by_compilation':False,'optimizer_or_MA_changed':False,
                'forward_changed':True,'forward_bitwise_parity_claimed':False,'module_lifetime':'process-owned',
                'reduction_schedule':'Contiguous Torch61 q-row warp; Torch263 vector4/head/tail, x32 rows>=16 elsex64; bitwise gate still required'}
            metadata_path.write_text(json.dumps(self.metadata,indent=2)+'\n',encoding='utf-8')
        if (self.path/'matrix_forward.cu').read_text(encoding='utf-8')!=generated or (self.path/'source_template.cu').read_bytes()!=SOURCE.read_bytes():
            raise RuntimeError('Forward fusion compiled-source copy changed')
        self.artifacts={name:sha(self.path/name) for name in ('build.json','matrix_forward.cu','source_template.cu','matrix_forward.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with self.torch.cuda.device(self.device):
            self.torch.cuda.current_stream(self.device)
            blob=C.create_string_buffer(ptx);self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            self.functions={}
            for name in ('matrix_forward_prepare','matrix_forward_dense'):
                fn=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
                self.functions[name]=fn
        runtime=self;t=self.torch
        class MatrixInputs(t.autograd.Function):
            @staticmethod
            def forward(ctx,indices,x):
                a,pi,sqrt_pi,live_outer,s,mr,rates=runtime.forward(indices,x)
                ctx.save_for_backward(s,a,pi,sqrt_pi,mr,rates,x,indices)
                ctx.mark_non_differentiable(pi,sqrt_pi,live_outer)
                return a,pi,sqrt_pi,live_outer
            @staticmethod
            def backward(ctx,g,*ignored):
                s,a,pi,sqrt_pi,mr,rates,x,indices=ctx.saved_tensors
                gradient=runtime.reverse.backward(g.contiguous(),s,a,pi,sqrt_pi,mr,rates,x,indices)
                return None,gradient
        self.matrix_inputs=MatrixInputs.apply
    def _bind(self,name,args):
        fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('CUDA driver operation failed: '+str(code))
    def launch(self,name,tensors,rows,extra=(),grid=None):
        arguments=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_int(rows)]+[C.c_int(v) for v in extra]
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        stream=self.torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.functions[name],rows if grid is None else grid,1,1,256,1,1,0,stream,pointers,None))
    def forward(self,indices,x):
        t=self.torch;rows=x.shape[0];owner=self.owner
        if rows<1 or rows>1024 or rows&(rows-1) or not x.is_contiguous() or not indices.is_contiguous():
            raise ValueError('Forward fusion supports positive power-of-two graph rows<=1024 and contiguous requests')
        pi=t.empty((rows,61),dtype=t.float64,device=self.device);sqrt_pi=t.empty_like(pi)
        mr=t.empty(rows,dtype=t.float64,device=self.device);rates=t.empty((rows,6),dtype=t.float64,device=self.device)
        diagonal=t.empty_like(pi);s=t.empty((rows,61,61),dtype=t.float64,device=self.device);a=t.empty_like(s)
        live=t.empty(s.shape,dtype=t.bool,device=self.device)
        self.launch('matrix_forward_prepare',[owner.pi,x,indices,owner.nkappa,owner.rate_map,pi,sqrt_pi,mr,rates,diagonal],
            rows,(owner.pi.stride(0),))
        self.launch('matrix_forward_dense',[pi,sqrt_pi,mr,rates,diagonal,x,indices,owner.nkappa,s,a,live],
            rows,grid=(rows*3721+255)//256)
        return a,pi,sqrt_pi,live,s,mr,rates
    def manifest(self):
        after={name:sha(self.path/name) for name in self.artifacts}
        if (after!=self.artifacts or sha(SOURCE)!=self.metadata['source_template_sha256']
            or sha(REDUCE_HEADER)!=self.metadata['torch_reduce_header_sha256']):
            raise RuntimeError('Forward fusion artifacts/source/header changed during run')
        return dict(self.metadata,build=str(self.path),reverse=self.reverse.manifest(),
            input_output='A,pi,sqrt_pi,live_outer exactly original interface; full unmasked mr',
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            requires_HP28_and_full_fit_gates=True)
