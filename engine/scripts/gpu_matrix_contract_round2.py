"""Native FP64 reverse Jacobian; forward stays the original Torch expression."""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
from pathlib import Path
from cuda_driver import compile_ptx
from torch_matrix_prototype import codon_edges
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/matrix_contract_v1.cu'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class MatrixContractRuntime:
    def __init__(self,owner):
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        if [tuple(v) for v in owner.prepared.edges]!=codon_edges():
            raise ValueError('Matrix Jacobian requires the exact code1 one-change edges')
        self.path=ROOT/'build/gpu-matrix-contract-round2-v1';self.path.mkdir(parents=True,exist_ok=True)
        metadata_path=self.path/'build.json';ptx_path=self.path/'matrix_contract.ptx'
        compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
        if metadata_path.exists():
            self.metadata=json.loads(metadata_path.read_text())
            if (self.metadata['source_sha256']!=sha(SOURCE) or self.metadata['options']!=OPTIONS
                or self.metadata['nvrtc_sha256']!=historical_compiler_sha(compiler)):
                raise RuntimeError('Preserve earlier matrix-contract build; source/options/compiler changed')
            ptx=ptx_path.read_bytes()
            if hashlib.sha256(ptx).hexdigest()!=self.metadata['ptx_sha256']:
                raise RuntimeError('Matrix-contract PTX changed')
        else:
            ptx,log=compile_ptx(SOURCE.read_text(encoding='utf-8'),OPTIONS)
            ptx_path.write_bytes(ptx);(self.path/'matrix_contract.cu').write_bytes(SOURCE.read_bytes())
            (self.path/'build.log').write_text(log,encoding='utf-8')
            self.metadata={'source_sha256':sha(SOURCE),'ptx_sha256':hashlib.sha256(ptx).hexdigest(),
                'nvrtc_sha256':historical_compiler_sha(compiler),'options':OPTIONS,'precision':'FP64 input/output/reduction/Jacobian',
                'edge_count':len(owner.prepared.edges),'CUDA_context_created_by_compilation':False,
                'fast_math':False,'forward_changed':False,'optimizer_or_MA_changed':False}
            metadata_path.write_text(json.dumps(self.metadata,indent=2)+'\n',encoding='utf-8')
        if (self.path/'matrix_contract.cu').read_bytes()!=SOURCE.read_bytes():
            raise RuntimeError('Matrix-contract compiled-source copy differs')
        self.artifacts={name:sha(self.path/name) for name in ('build.json','matrix_contract.cu','matrix_contract.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        self._bind('cuModuleUnload',[C.c_void_p])
        with self.torch.cuda.device(self.device):
            self.torch.cuda.current_stream(self.device)
            blob=C.create_string_buffer(ptx);self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            self.functions={}
            for name in ('matrix_contract_reduce','matrix_contract_edges'):
                fn=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
                self.functions[name]=fn
        runtime=self;t=self.torch
        class MatrixInputs(t.autograd.Function):
            @staticmethod
            def forward(ctx,indices,x):
                # Verbatim statement order from publication_gpu_likelihood.
                pi=owner.pi[indices,:61]
                omega=x.gather(1,(owner.nkappa[indices]+1)[:,None])[:,0]
                raw=t.where(owner.active_parameter[indices],x[:,1:7],t.ones_like(x[:,1:7]))
                rates=raw.gather(1,owner.rate_map[indices]);q,mr=owner._q_mr(rates,pi,omega)
                live=pi>0.;sqrt_pi=t.where(live,pi,t.ones_like(pi)).sqrt()
                s=(q*sqrt_pi[:,:,None])/sqrt_pi[:,None,:]
                lower=t.tril(s,diagonal=-1)
                symmetric=lower+lower.transpose(-1,-2)+t.diag_embed(q.diagonal(dim1=-2,dim2=-1))
                live_outer=live[:,:,None]&live[:,None,:]
                symmetric=t.where(live_outer,symmetric,t.zeros_like(symmetric))
                a=(symmetric*x[:,0,None,None])/mr[:,None,None]
                ctx.save_for_backward(symmetric,a,pi,sqrt_pi,mr,rates,x,indices)
                ctx.mark_non_differentiable(pi,sqrt_pi,live_outer)
                return a,pi,sqrt_pi,live_outer
            @staticmethod
            def backward(ctx,g,*ignored):
                s,a,pi,sqrt_pi,mr,rates,x,indices=ctx.saved_tensors
                gradient=runtime.backward(g.contiguous(),s,a,pi,sqrt_pi,mr,rates,x,indices)
                return None,gradient
        self.matrix_inputs=MatrixInputs.apply

    def _bind(self,name,args):
        fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=C.c_int
    def _check(self,code):
        if code:raise RuntimeError('CUDA driver operation failed: '+str(code))
    def launch(self,name,tensors,rows,extra=()):
        arguments=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_int(rows)]+[C.c_int(v) for v in extra]
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        stream=self.torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.functions[name],rows,1,1,256,1,1,0,stream,pointers,None))
    def backward(self,g,s,a,pi,sqrt_pi,mr,rates,x,indices):
        rows=x.shape[0];t=self.torch;owner=self.owner
        scratch=t.empty((rows,2),dtype=t.float64,device=self.device)
        gradient=t.empty((rows,7),dtype=t.float64,device=self.device)
        self.launch('matrix_contract_reduce',[g,s,a,mr,scratch],rows)
        self.launch('matrix_contract_edges',[g,pi,sqrt_pi,mr,rates,x,indices,owner.nkappa,owner.rate_map,
            owner.edge_i,owner.edge_j,owner.edge_rate,owner.edge_syn,scratch,gradient],rows,(len(owner.prepared.edges),))
        return gradient
    def manifest(self):
        after={name:sha(self.path/name) for name in self.artifacts}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256']:
            raise RuntimeError('Matrix-contract artifacts/source changed during run')
        return dict(self.metadata,build=str(self.path),reduction='256-thread per row, fixed binary tree',
            reverse='All 263 codon edges; masked S, full unmasked mr; time-zero uses S/mr directly',
            original_forward='Verbatim publication_gpu_likelihood statement order; no autograd tape',
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            gradient_bit_parity_claimed=False,requires_HP28_and_full_fit_gates=True)
