"""Exact prepared-support first-adjoint products, full dense downstream values.

Importing this file does not import Torch, load CUDA, or compile. This runtime
is a new candidate requiring root-owned primitive, HP28, partition and full-fit
tests. It is valid only for the objective's first adjoint; later adjoints are
not sparse. The GPU certificate verifies the support/finite precondition before
each product, and every unsafe row executes the complete original k0..63 FMA.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha

from array import array
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path

from cuda_driver import compile_ptx
from gpu_register_gemm_round5 import validate_operands

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round6/observed_adjoint_products_v1.cu'
RUNTIME_SOURCE=Path(__file__).resolve()
BUILD=ROOT/'build/gpu-observed-adjoint-round6-fp64-v1'
ORIGINAL_SOURCE=ROOT/'src/gpu_round2/gemm_coalesced_fp64_v1.cu'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
ARTIFACTS=('build.json','source_template.cu','nvrtc_input.cu','observed_adjoint.ptx')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def observed_support(prepared):
    """Exact original objective support; logical g coordinates never transpose."""
    count=len(prepared.z0)
    if not count or len(prepared.z1)!=count or len(prepared.weights)!=count:
        raise ValueError('Prepared pattern arrays differ')
    csr_ptr=[];csr_col=[];csc_ptr=[];csc_row=[];masks=[];counts=[]
    for z0,z1,weights in zip(prepared.z0,prepared.z1,prepared.weights):
        if len(z0)!=len(z1) or len(z0)!=len(weights):
            raise ValueError('Prepared pattern length mismatch')
        coordinates=[]
        for i,j,weight in zip(z0,z1,weights):
            if not isinstance(i,int) or not isinstance(j,int) or not 0<=i<61 or not 0<=j<61:
                raise ValueError('Prepared pattern code outside0..60')
            if not math.isfinite(weight) or weight<0:
                raise ValueError('Invalid prepared nonnegative weight')
            if weight>=1e-6:
                coordinates.append((i,j))
        if len(set(coordinates))!=len(coordinates):
            raise ValueError('Duplicate observed dense-weight cell')
        mask=[False]*3721
        rows=[[] for _ in range(61)];cols=[[] for _ in range(61)]
        for i,j in coordinates:
            mask[i*61+j]=True;rows[i].append(j);cols[j].append(i)
        row_ptr=[len(csr_col)];col_ptr=[len(csc_row)]
        for r in rows:
            csr_col.extend(sorted(r));row_ptr.append(len(csr_col))
        for c in cols:
            csc_row.extend(sorted(c));col_ptr.append(len(csc_row))
        csr_ptr.append(row_ptr);csc_ptr.append(col_ptr);masks.append(mask);counts.append(len(coordinates))
    # Upload of an empty global CSR/CSC allocation is unnecessary and rejected.
    if not csr_col:
        raise ValueError('Prepared objective support is globally empty')
    encoded=(array('i',[v for row in csr_ptr for v in row]).tobytes()+array('i',csr_col).tobytes()
             +array('i',[v for row in csc_ptr for v in row]).tobytes()+array('i',csc_row).tobytes()
             +bytes(v for row in masks for v in row))
    return {'csr_ptr':csr_ptr,'csr_col':csr_col,'csc_ptr':csc_ptr,'csc_row':csc_row,'mask':masks,
            'counts':counts,'SHA256':hashlib.sha256(encoded).hexdigest(),'pairs':count}

def build_observed_adjoint(allow_compile=False):
    """CPU NVRTC build only; no CUDA Driver/device/context initialization."""
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    source=SOURCE.read_text(encoding='utf-8')
    identity={'source_sha256':sha(SOURCE),'NVRTC_input_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'runtime_source_sha256':sha(RUNTIME_SOURCE),
              'original_source_sha256':sha(ORIGINAL_SOURCE),'options':OPTIONS,
              'nvrtc_sha256':historical_compiler_sha(compiler),'nvrtc_builtins_sha256':historical_compiler_sha(builtins)}
    BUILD.mkdir(parents=True,exist_ok=True)
    record=BUILD/'build.json';ptx_path=BUILD/'observed_adjoint.ptx'
    if record.exists():
        metadata=json.loads(record.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()):
            raise RuntimeError('Preserve observed-adjoint build; identity changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:
            raise RuntimeError('Observed-adjoint PTX changed')
    else:
        if not allow_compile:
            raise RuntimeError('Cached observed-adjoint PTX required; run CPU-only --build first')
        if any(BUILD.iterdir()):
            raise RuntimeError('Preserve incomplete observed-adjoint build')
        ptx,log=compile_ptx(source,OPTIONS)
        ptx_path.write_bytes(ptx);(BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
        (BUILD/'nvrtc_input.cu').write_bytes(source.encode());(BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            precision='FP64 inputs/output/accumulator; no fast-math/TF32',
            arithmetic='Ascending retained original k __fma_rn; positive-zero tail normalization; unsafe rows original k0..63',
            safety='All logical g/dense cells finite and every off-support g cell equals zero, checked on GPU for every product row',
            output='Full dense61x61; never masks downstream adjoints',
            CUDA_context_created_by_compilation=False,optimizer_or_MA_changed=False,precision_change=False)
        record.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes() or (BUILD/'nvrtc_input.cu').read_bytes()!=source.encode():
        raise RuntimeError('Observed-adjoint compiled source copies changed')
    return ptx,metadata

class ObservedAdjointRuntime:
    def __init__(self,api,device,prepared,force_full=False):
        if type(force_full) is not bool:
            raise ValueError('force_full must be explicit bool')
        self.force_full=force_full
        self.runtime_source_before=sha(RUNTIME_SOURCE)
        self.torch,self.device=api,api.device(device)
        if self.device.type!='cuda' or self.device.index is None:
            raise ValueError('ObservedAdjointRuntime requires explicit single CUDA device')
        host=observed_support(prepared)
        self.support_metadata={'SHA256':host['SHA256'],'pairs':host['pairs'],
            'nonzero_pattern_counts':host['counts'],'slot_mapping':'pair=slot//14; all14models retained',
            'support':'Exact prepared z0,z1 cells with weight>=1e-6, no state cutoff'}
        self.pairs=host['pairs']
        self.csr_ptr=api.tensor(host['csr_ptr'],dtype=api.int32,device=self.device)
        self.csr_col=api.tensor(host['csr_col'],dtype=api.int32,device=self.device)
        self.csc_ptr=api.tensor(host['csc_ptr'],dtype=api.int32,device=self.device)
        self.csc_row=api.tensor(host['csc_row'],dtype=api.int32,device=self.device)
        self.support=api.tensor(host['mask'],dtype=api.bool,device=self.device)
        ptx,self.metadata=build_observed_adjoint()
        self.artifacts={n:sha(BUILD/n) for n in ARTIFACTS}
        self.driver=cuda_driver();self.module=C.c_void_p();self.functions={}
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with api.cuda.device(self.device):
            api.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            try:
                for name in ('observed_adjoint_safe','observed_adjoint_left','observed_adjoint_right'):
                    handle=C.c_void_p()
                    self._check(self.driver.cuModuleGetFunction(C.byref(handle),self.module,name.encode()))
                    self.functions[name]=handle
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module)
                self.module=C.c_void_p()
                if cleanup:raise RuntimeError('Observed-adjoint lookup and partial module cleanup failed: '+str(cleanup)) from error
                raise

    def _bind(self,name,args):
        fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=C.c_int

    def _check(self,code):
        if code:raise RuntimeError('Observed-adjoint CUDA Driver operation failed: '+str(code))

    def _launch(self,name,arguments,grid,block):
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(v),C.c_void_p) for v in arguments))
        stream=self.torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.functions[name],*grid,*block,0,stream,pointers,None))

    def _validate(self,g,dense,indices):
        validate_operands(self.torch,self.device,g,dense)
        if (indices.ndim!=1 or indices.shape!=(g.shape[0],) or indices.dtype!=self.torch.int64
            or indices.device!=self.device or indices.stride(0)<0):
            raise ValueError('Observed-adjoint requires same-row CUDA int64 slot indices with nonnegative stride')

    def safety(self,g,dense,indices):
        self._validate(g,dense,indices)
        safe=self.torch.empty((g.shape[0],),dtype=self.torch.int32,device=self.device)
        arguments=[C.c_uint64(x.data_ptr()) for x in (g,dense,indices,self.support,safe)]
        arguments.extend(C.c_int64(s) for s in (*g.stride(),*dense.stride(),indices.stride(0)))
        arguments.extend((C.c_int(g.shape[0]),C.c_int(self.pairs),C.c_int(self.force_full)))
        self._launch('observed_adjoint_safe',arguments,(g.shape[0],1,1),(256,1,1))
        return safe

    def _product(self,g,dense,indices,right):
        self._validate(g,dense,indices)
        safe=self.safety(g,dense,indices)
        out=self.torch.empty(g.shape,dtype=self.torch.float64,device=self.device)
        ptr,col=(self.csc_ptr,self.csc_row) if right else (self.csr_ptr,self.csr_col)
        arguments=[C.c_uint64(x.data_ptr()) for x in (g,dense,indices,ptr,col,safe,out)]
        arguments.extend(C.c_int64(s) for s in (*g.stride(),*dense.stride(),indices.stride(0)))
        arguments.append(C.c_int(g.shape[0]))
        self._launch('observed_adjoint_right' if right else 'observed_adjoint_left',arguments,(2,16,g.shape[0]),(32,4,1))
        return out

    def product_left(self,g,dense,indices):
        """Full dense result g @ dense; g uses original objective coordinates."""
        return self._product(g,dense,indices,False)

    def product_right(self,g,dense,indices):
        """Full dense result dense @ g; g uses original objective coordinates."""
        return self._product(g,dense,indices,True)

    def close(self):
        """Caller must drain owned streams/graphs before unloading the module."""
        if self.module.value:
            self._check(self.driver.cuModuleUnload(self.module));self.module=C.c_void_p()

    def manifest(self):
        after={n:sha(BUILD/n) for n in ARTIFACTS}
        if (after!=self.artifacts or sha(SOURCE)!=self.metadata['source_sha256']
            or sha(ORIGINAL_SOURCE)!=self.metadata['original_source_sha256']
            or sha(RUNTIME_SOURCE)!=self.runtime_source_before):
            raise RuntimeError('Observed-adjoint source/build changed')
        return dict(self.metadata,schema='kaksforge.observed_first_adjoint_round6/1',build=str(BUILD),
            prepared_support=self.support_metadata,artifact_SHA256_before=self.artifacts,
            artifact_SHA256_after=after,artifact_unchanged=True,
            scope='Only first objective adjoint contractions; all outputs and later adjoints remain dense',
            actual_bytewise_parity_requires_GPU_gate=True,
            row_safety_checked_every_product=True,all14_models_preserved=True,
            forced_full_fallback=self.force_full,runtime_source_unchanged=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.build:
        ptx,metadata=build_observed_adjoint(allow_compile=True)
        print(json.dumps({'PASS':True,'build':str(BUILD),'ptx_bytes':len(ptx),
                          'ptx_sha256':metadata['ptx_sha256'],'CUDA_context_created':False}))
    else:
        print(json.dumps({'source':str(SOURCE),'source_sha256':sha(SOURCE),
                          'CUDA_initialized':False,'cached_build_exists':(BUILD/'build.json').is_file()}))

if __name__=='__main__':
    main()
