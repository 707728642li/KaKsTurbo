"""FP64 final-Jacobian-only selected VJP; no upstream support masking.

Import/static checks and --build create no CUDA context. The caller owns every
GPU launch and freezes/gates this candidate before any benchmark claim.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from product_v1_cached_assets import historical_compiler_sha

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

from cuda_driver import compile_ptx
from gpu_sparse_t18_round2 import structural_csr
from torch_matrix_prototype import codon_edges

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/gpu_round6/selected_final_vjp_v1.cu'
RUNTIME_SOURCE=Path(__file__).resolve()
ORIGINAL_REVERSE=ROOT/'src/gpu_round2/matrix_contract_v1.cu'
ORIGINAL_DENSE=ROOT/'src/gpu_round2/gemm_coalesced_fp64_v1.cu'
ORIGINAL_SPARSE=ROOT/'src/gpu_round2/sparse_t18_v1.cu'
ORIGINAL_HELPERS=ROOT/'src/gpu_round2/native_t18_intscale_helpers_v1.cu'
BUILD=ROOT/'build/gpu-selected-final-vjp-round6-fp64-v1'
OPTIONS=['--std=c++17','--gpu-architecture=compute_86','--fmad=false']
KERNELS=('selected_row_certificate','selected_tail_packed','selected_tail_full_fallback',
         'selected_contract_reduce','selected_contract_edges')
ARTIFACTS=('build.json','selected_final_vjp.cu','source_template.cu',
           'original_matrix_contract.cu','structural_map.json','selected_final_vjp.ptx')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structural_map():
    rp,col=structural_csr()
    targets=[(i,col[p]) for i in range(61) for p in range(rp[i],rp[i+1])]
    lookup=[-1]*3721
    for ordinal,(i,j) in enumerate(targets):lookup[i*61+j]=ordinal
    if len(targets)!=587 or len(set(targets))!=587 or any(lookup[i*61+i]<0 for i in range(61)):
        raise RuntimeError('Complete all263edge+61diag support required')
    return dict(N=61,full_entries=3721,selected_entries=587,undirected_edges=263,
        row_pointers=rp,column_indices=col,target_i=[p[0] for p in targets],
        target_j=[p[1] for p in targets],lookup=lookup,degree_min=min(rp[i+1]-rp[i] for i in range(61)),
        degree_max=max(rp[i+1]-rp[i] for i in range(61)),
        degree_squared_sum=sum((rp[i+1]-rp[i])**2 for i in range(61)))


def edge_source():
    old=ORIGINAL_REVERSE.read_text(encoding='utf-8')
    marker='extern "C" __global__ void matrix_contract_edges('
    if old.count(marker)!=1:raise RuntimeError('Original all-edge entrypoint changed')
    edge=old[old.index(marker):]
    expected='const double* gr=g+(long long)row*3721;'
    if edge.count(expected)!=1:raise RuntimeError('Original all-edge gradient pointer changed')
    edge=edge.replace('matrix_contract_edges(const double* g,',
        'selected_contract_edges(const double* g,const double* full,const int* safe,',1)
    edge=edge.replace(expected,
        'const bool packed=safe[row]!=0;\n    const double* gr=packed?g+(long long)row*587:full+(long long)row*3721;',1)
    for cell in ('i*61+j','j*61+i','i*61+i','j*61+j'):
        edge=edge.replace('gr['+cell+']','gr[packed?LOOKUP['+cell+']:'+cell+']')
    return edge


def kernel_source():
    mapping=structural_map()
    text=SOURCE.read_text(encoding='utf-8')
    replacements={'ROW_POINTER_VALUES':mapping['row_pointers'],'COLUMN_VALUES':mapping['column_indices'],
        'TARGET_I_VALUES':mapping['target_i'],'TARGET_J_VALUES':mapping['target_j'],
        'LOOKUP_VALUES':mapping['lookup']}
    for marker,values in replacements.items():
        if text.count(marker)!=1:raise RuntimeError('Selected VJP template marker changed: '+marker)
        text=text.replace(marker,','.join(map(str,values)))
    if text.count('ORIGINAL_EDGE_KERNEL')!=1:raise RuntimeError('Selected edge insertion changed')
    return text.replace('ORIGINAL_EDGE_KERNEL',edge_source())


def static_contract():
    mapping=structural_map();text=SOURCE.read_text(encoding='utf-8');edge=edge_source()
    if mapping['degree_min']!=8 or mapping['degree_max']!=10 or mapping['degree_squared_sum']!=5673:
        raise RuntimeError('Frozen code1 CSR arithmetic counts changed')
    old=ORIGINAL_REVERSE.read_text(encoding='utf-8');marker='extern "C" __global__ void matrix_contract_edges('
    normalized=edge.replace('selected_contract_edges(const double* g,const double* full,const int* safe,',
        'matrix_contract_edges(const double* g,',1).replace(
        'const bool packed=safe[row]!=0;\n    const double* gr=packed?g+(long long)row*587:full+(long long)row*3721;',
        'const double* gr=g+(long long)row*3721;',1)
    for cell in ('i*61+j','j*61+i','i*61+i','j*61+j'):
        normalized=normalized.replace('gr[packed?LOOKUP['+cell+']:'+cell+']','gr['+cell+']')
    if normalized!=old[old.index(marker):]:raise RuntimeError('All-edge algebra changed beyond gradient lookup')
    for statement in ('for(int k=0;k<64;k++)','dense=__fma_rn(left,right,dense);',
        'double gz_updated=gz[off+i*N+j]+dense;',
        'return ((gz_updated+right_product)+left_product)*scale;',
        'for(int p=tid;p<NN;p+=256)', 'safe[row]=!force_full && !invalid[0] && isfinite(bound);'):
        if statement not in text:raise RuntimeError('Selected VJP contract statement changed: '+statement)
    return dict(PASS=True,full_upstream_adjoints_required=True,full_ga3_and_updated_ga2=True,
        support='All61diagonal+bothdirectionsall263structuraledges; no pi or observedpattern masking',
        full_unmasked_mean_rate_derivative=True,all263_edge_algebra_unchanged_except_lookup=True,
        ascending_dense64_FMA=True,original_CSR_order=True,original_add_scale_order=True,
        original_full3721_logical_reduction_order=True,original_binary_tree=True,
        safe_row='All5fulltailinputsfinite,S/Afinite,nonedgesexactpositive0,upwardroundedoverflowcertificate',
        unsafe_row='All3721tailentriescomputed; originalfullcontraction/edge algebra retained',
        original_forward_unchanged=True,optimizer_models_scaling_bounds_unchanged=True,
        selected_tail_bits_require_GPU_gate=True,whole_gradient_bit_parity_claimed=False,
        signed_zero_subnormal_nonfinite_and_forced_fallback_require_actual_GPU_gates=True,
        selected_entries=mapping['selected_entries'],degree_squared_sum=mapping['degree_squared_sum'])


def build_selected_vjp(allow_compile=False):
    generated=kernel_source();mapping=structural_map();contract=static_contract()
    map_bytes=(json.dumps(mapping,sort_keys=True,separators=(',',':'))+'\n').encode()
    compiler=ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'
    builtins=ROOT/'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll'
    identity=dict(source_template_sha256=sha(SOURCE),generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        runtime_source_sha256=sha(RUNTIME_SOURCE),original_reverse_sha256=sha(ORIGINAL_REVERSE),
        original_dense_sha256=sha(ORIGINAL_DENSE),original_sparse_sha256=sha(ORIGINAL_SPARSE),
        original_helpers_sha256=sha(ORIGINAL_HELPERS),structural_map_sha256=hashlib.sha256(map_bytes).hexdigest(),
        options=OPTIONS,nvrtc_sha256=historical_compiler_sha(compiler),nvrtc_builtins_sha256=historical_compiler_sha(builtins),static_contract=contract)
    metadata_path=BUILD/'build.json';ptx_path=BUILD/'selected_final_vjp.ptx'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k)!=v for k,v in identity.items()) or sha(ptx_path)!=metadata['ptx_sha256']:
            raise RuntimeError('Preserve selected VJP build; source/compiler/PTX identity changed')
        if (BUILD/'selected_final_vjp.cu').read_bytes()!=generated.encode() or (BUILD/'source_template.cu').read_bytes()!=SOURCE.read_bytes() or (BUILD/'structural_map.json').read_bytes()!=map_bytes or (BUILD/'original_matrix_contract.cu').read_bytes()!=ORIGINAL_REVERSE.read_bytes():
            raise RuntimeError('Selected VJP build snapshots changed')
        return ptx_path.read_bytes(),metadata
    if not allow_compile:raise RuntimeError('Precompiled selected VJP PTX required; run --build before GPU execution')
    if BUILD.exists() and any(BUILD.iterdir()):raise RuntimeError('Preserve incomplete selected VJP build')
    BUILD.mkdir(parents=True,exist_ok=True)
    (BUILD/'selected_final_vjp.cu').write_bytes(generated.encode())
    (BUILD/'source_template.cu').write_bytes(SOURCE.read_bytes())
    (BUILD/'original_matrix_contract.cu').write_bytes(ORIGINAL_REVERSE.read_bytes())
    (BUILD/'structural_map.json').write_bytes(map_bytes)
    try:ptx,log=compile_ptx(generated,OPTIONS)
    except BaseException as error:
        (BUILD/'compile_failure.json').write_text(json.dumps(dict(identity,error_type=type(error).__name__,error=str(error),CUDA_context_created=False),indent=2)+'\n',encoding='utf-8')
        raise
    if sha(SOURCE)!=identity['source_template_sha256'] or sha(RUNTIME_SOURCE)!=identity['runtime_source_sha256'] or sha(ORIGINAL_REVERSE)!=identity['original_reverse_sha256']:
        raise RuntimeError('Selected VJP sources changed during CPU NVRTC compilation')
    ptx_path.write_bytes(ptx);(BUILD/'build.log').write_text(log,encoding='utf-8')
    metadata=dict(identity,schema='kaksforge.selected_final_VJP_FP64_build_round6/1',
        ptx_sha256=hashlib.sha256(ptx).hexdigest(),CUDA_context_created_by_compilation=False,
        precision='FP64 FMA64/CSR/order/exactscale; fmadfalse/no fastmath',
        selected_only_at_final_parameter_Jacobian=True,forward_and_upstream_dense_tape_changed=False)
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return ptx,metadata


class SelectedVJPRuntime:
    def __init__(self,owner,original_reverse,force_full=False):
        if type(force_full) is not bool:raise ValueError('force_full must be explicit bool')
        if getattr(original_reverse,'owner',None) is not owner:raise ValueError('Original reverse belongs to a different provider')
        if [tuple(v) for v in owner.prepared.edges]!=codon_edges():raise ValueError('All263 original code1 edges required')
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        self.force_full=force_full;self.path=BUILD
        self.original_reverse_manifest=original_reverse.manifest()
        ptx,self.metadata=build_selected_vjp()
        self.runtime_source_before=sha(RUNTIME_SOURCE)
        self.artifacts={name:sha(BUILD/name) for name in ARTIFACTS}
        self.driver=cuda_driver();self.module=C.c_void_p()
        self._bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind('cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        self._bind('cuModuleUnload',[C.c_void_p])
        with self.torch.cuda.device(self.device):
            self.torch.cuda.current_stream(self.device);blob=C.create_string_buffer(ptx)
            self._check(self.driver.cuModuleLoadData(C.byref(self.module),blob))
            self.functions={}
            try:
                for name in KERNELS:
                    function=C.c_void_p()
                    self._check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()))
                    self.functions[name]=function
            except BaseException as error:
                cleanup=self.driver.cuModuleUnload(self.module);self.module=C.c_void_p()
                if cleanup:raise RuntimeError('Selected VJP lookup and partial unload failed: '+str(cleanup)) from error
                raise
        self.last_row_safety=self.last_packed=self.last_full=None

    def _bind(self,name,args):
        function=getattr(self.driver,name);function.argtypes=args;function.restype=C.c_int

    @staticmethod
    def _check(code):
        if code:raise RuntimeError('CUDA selected final VJP operation failed: '+str(code))

    def launch(self,name,tensors,rows,extra=(),grid=None):
        arguments=[C.c_uint64(value.data_ptr()) for value in tensors]+[C.c_int(rows)]+[C.c_int(value) for value in extra]
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.byref(value),C.c_void_p) for value in arguments))
        shape=(rows,1,1) if grid is None else grid
        stream=self.torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.functions[name],*shape,256,1,1,0,stream,pointers,None))

    def finish(self,fold_gz,ga3,a2,updated_ga2,z,diagnostic,s,a,pi,sqrt_pi,mr,rates,x,indices):
        t=self.torch;rows=x.shape[0];owner=self.owner
        if not 1<=rows<=1024:raise ValueError('Selected tail requires1..1024rows')
        for value in (fold_gz,ga3,a2,updated_ga2,z,s,a):
            if value.shape!=(rows,61,61) or value.dtype!=t.float64 or value.device!=self.device or not value.is_contiguous():
                raise ValueError('Selected tail fullinput must be same-device contiguous FP64[rows,61,61]')
        for value,shape in ((diagnostic,(rows,4)),(pi,(rows,61)),(sqrt_pi,(rows,61)),(mr,(rows,)),(rates,(rows,6)),(x,(rows,7))):
            if value.shape!=shape or value.dtype!=t.float64 or value.device!=self.device or not value.is_contiguous():
                raise ValueError('Selected tail metadata FP64 layout differs')
        if indices.shape!=(rows,) or indices.dtype!=t.int64 or indices.device!=self.device or not indices.is_contiguous():
            raise ValueError('Selected tail indices require contiguous selected-device int64')
        safe=t.empty((rows,),dtype=t.int32,device=self.device)
        packed=t.empty((rows,587),dtype=t.float64,device=self.device)
        full=t.empty((rows,61,61),dtype=t.float64,device=self.device)
        scratch=t.empty((rows,2),dtype=t.float64,device=self.device)
        raw=t.empty((rows,7),dtype=t.float64,device=self.device)
        self.launch('selected_row_certificate',[fold_gz,ga3,a2,updated_ga2,z,s,a,safe],rows,(int(self.force_full),))
        tail=[fold_gz,ga3,a2,updated_ga2,z,diagnostic,safe]
        self.launch('selected_tail_packed',[*tail,packed],rows,grid=((587+255)//256,rows,1))
        self.launch('selected_tail_full_fallback',[*tail,full],rows,grid=((3721+255)//256,rows,1))
        self.launch('selected_contract_reduce',[packed,full,safe,s,a,mr,scratch],rows)
        self.launch('selected_contract_edges',[packed,full,safe,pi,sqrt_pi,mr,rates,x,indices,owner.nkappa,owner.rate_map,
            owner.edge_i,owner.edge_j,owner.edge_rate,owner.edge_syn,scratch,raw],rows,(len(owner.prepared.edges),))
        self.last_row_safety,self.last_packed,self.last_full=safe,packed,full
        return raw

    def manifest(self):
        after={name:sha(BUILD/name) for name in self.artifacts}
        if after!=self.artifacts or sha(RUNTIME_SOURCE)!=self.runtime_source_before or sha(SOURCE)!=self.metadata['source_template_sha256']:
            raise RuntimeError('Selected VJP source/artifacts changed during execution')
        static_contract()
        return dict(self.metadata,build=str(BUILD),runtime_source_unchanged=True,
            artifact_SHA256_before=self.artifacts,artifact_SHA256_after=after,artifact_unchanged=True,
            original_reverse_provenance=self.original_reverse_manifest,
            API='finish(fold_gz,ga3,a2,updated_ga2,z,diagnostic,s,a,pi,sqrt_pi,mr,rates,x,indices)->raw7',
            variant='naive-selected-output-FMA64',forced_full_fallback=self.force_full,
            original_dense_upstream_adjoints_retained=True,selected_only_final_observable_Jacobian_coordinates=True,
            all14_models_full263_edges_and_mr=True,CUDA_module_lifetime='provider-owned; root installer unloads replaced original reverse',
            no_GPU_benchmark_or_correctness_pass_claimed=True,requires_HP28_forced_fallback_partition_and_fullfit_gates=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.build:
        ptx,metadata=build_selected_vjp(allow_compile=True)
        print(json.dumps(dict(PASS=True,build=str(BUILD),ptx_bytes=len(ptx),ptx_sha256=metadata['ptx_sha256'],CUDA_context_created=False)))
    else:print(json.dumps(static_contract()))


if __name__=='__main__':main()
