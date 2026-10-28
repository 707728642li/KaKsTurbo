"""Isolated exact structural-zero shortcut in native matrix dense emission.

Original forward preparation, full mr and MatrixContract reverse are inherited.
Only fixed offdiagonal nonedges with finite positive time/mr and valid support
may directly emit +0; every failed precondition falls through original code.
No incoming adjoints are masked. No Torch autograd or GPU at import/build time.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha
import ctypes as C
import hashlib
import json
from pathlib import Path
from cuda_driver import compile_ptx
from gpu_forward_matrix_fusion_round2 import ForwardMatrixFusionRuntime,REDUCE_HEADER,SOURCE as ORIGINAL_SOURCE
from gpu_matrix_contract_round2 import MatrixContractRuntime
from torch_matrix_prototype import codon_edges
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round2/matrix_forward_structural_zero_v1.cu'
BUILD=ROOT/'build/gpu-matrix-structural-zero-round2-fp64-v1'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
RUNTIME_SOURCE=Path(__file__).resolve()
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
def build_structural_matrix_ptx():
    generated=kernel_source();compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    identity=dict(source_template_sha256=sha(SOURCE),generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        original_template_sha256=sha(ORIGINAL_SOURCE),options=OPTIONS,nvrtc_sha256=historical_compiler_sha(compiler),
        nvrtc_builtins_sha256=historical_compiler_sha(builtins),torch_reduce_header_sha256=sha(REDUCE_HEADER))
    BUILD.mkdir(parents=True,exist_ok=True);metadata_path=BUILD/'build.json';ptx_path=BUILD/'matrix_structural.ptx'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()):raise RuntimeError('Preserve prior structural matrix build; source/compiler/options changed')
        ptx=ptx_path.read_bytes()
        if hashlib.sha256(ptx).hexdigest()!=metadata['ptx_sha256']:raise RuntimeError('Structural matrix PTX changed')
    else:
        if any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete structural matrix build; choose a new version')
        try:ptx,log=compile_ptx(generated,OPTIONS)
        except BaseException as error:
            (BUILD/'compile_failure.json').write_text(json.dumps(dict(identity,error_type=type(error).__name__,error=str(error),CUDA_context_created=False),indent=2)+'\n',encoding='utf-8')
            raise
        if sha(SOURCE)!=identity['source_template_sha256'] or sha(ORIGINAL_SOURCE)!=identity['original_template_sha256'] or historical_compiler_sha(compiler)!=identity['nvrtc_sha256'] or historical_compiler_sha(builtins)!=identity['nvrtc_builtins_sha256']:
            raise RuntimeError('Structural matrix source/compiler changed during compilation')
        ptx_path.write_bytes(ptx);(BUILD/'matrix_structural.cu').write_bytes(generated.encode())
        (BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes());(BUILD/'build.log').write_text(log,encoding='utf-8')
        metadata=dict(identity,schema='kaksforge.matrix_structural_zero_FP64_build_round2/1',ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            precision='Original FP64 arithmetic and positive exact-zero shortcut; no fastmath/TF32',
            shortcut='Offdiagonal CODE<0 with finite positive time/mr and finite nonnegative pi/positive sqrt; absent pi requires sqrt1',
            failure_time0_and_diagonal_fallthrough='Original arithmetic unchanged, including Inf/NaN/mr0/signedzero/time0',
            structure_cells=587,structural_offdiagonal_nonedges=3134,full_unmasked_mr_unchanged=True,
            reverse='Original MatrixContractRuntime; no adjoint masking',CUDA_context_created_by_compilation=False,
            optimizer_or_MA_changed=False,module_lifetime='process-owned',actual_GPU_bitparity_gate_required=True)
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if (BUILD/'matrix_structural.cu').read_bytes()!=generated.encode() or (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes():
        raise RuntimeError('Structural matrix compiled source copy changed')
    return ptx,metadata
class StructuralZeroForwardMatrixRuntime(ForwardMatrixFusionRuntime):
    def __init__(self,owner):
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        if [tuple(v) for v in owner.prepared.edges]!=codon_edges():raise ValueError('Structural matrix requires exact code1 edges')
        self.reverse=MatrixContractRuntime(owner);ptx,self.metadata=build_structural_matrix_ptx();self.path=BUILD
        self.runtime_source_sha256=sha(RUNTIME_SOURCE)
        self.artifacts={name:sha(BUILD/name) for name in ('build.json','matrix_structural.cu','source_template.cu','matrix_structural.ptx')}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p]);self._bind('cuModuleUnload',[C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        with self.torch.cuda.device(self.device):
            self.torch.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob));self.functions={}
            try:
                for name in ('matrix_forward_prepare','matrix_forward_dense'):
                    fn=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(fn),self.module,name.encode()));self.functions[name]=fn
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module)
                if cleanup:raise RuntimeError('Structural matrix lookup and partial module cleanup failed: '+str(cleanup)) from error
                self.module=C.c_void_p();raise
    def manifest(self):
        after={name:sha(self.path/name) for name in self.artifacts}
        if after!=self.artifacts or sha(SOURCE)!=self.metadata['source_template_sha256'] or sha(ORIGINAL_SOURCE)!=self.metadata['original_template_sha256'] or sha(REDUCE_HEADER)!=self.metadata['torch_reduce_header_sha256'] or sha(RUNTIME_SOURCE)!=self.runtime_source_sha256:
            raise RuntimeError('Structural matrix artifacts/source changed')
        return dict(self.metadata,build=str(self.path),reverse=self.reverse.manifest(),
            runtime_source_sha256=self.runtime_source_sha256,runtime_source_unchanged=True,
            input_output='Original A/pi/sqrt/live/S/mr/rates interface; forward/launch methods inherited exactly',
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            requires_HP28_and_full_fit_gates=True)
