"""CPU-only FP64 shared-model objective provider; no fitting or GPU imports.

Pairs are copied into the backend's bundled shared core. A session owns a
persistent thread pool with one reusable arena per worker. Every batch call
is synchronous and preserves input row order, including repeated slots.
"""
from __future__ import annotations
from product_v1_platform import blas_runtime, build_record, dll_directory, linux_blas_modules, native_path
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from shared_core import SharedCore

ROOT=Path(__file__).resolve().parents[1]
DIAGNOSTICS=('norm1','squarings','adaptive_fallback','gemm_count','minimum_observed_joint_probability',
             'active_row_sum_error','minimum_transition_entry','MKL_threads')

def mkl_modules():
    """Actually loaded MKL implementation/OpenMP DLLs, not only the dispatcher."""
    if os.name!='nt':return linux_blas_modules(ROOT)
    kernel=C.WinDLL('kernel32',use_last_error=True);psapi=C.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=C.c_void_p
    kernel.GetModuleFileNameW.argtypes=[C.c_void_p,C.c_wchar_p,C.c_uint32];kernel.GetModuleFileNameW.restype=C.c_uint32
    psapi.EnumProcessModules.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.c_uint32,C.POINTER(C.c_uint32)];psapi.EnumProcessModules.restype=C.c_int
    modules=(C.c_void_p*1024)();needed=C.c_uint32()
    if not psapi.EnumProcessModules(kernel.GetCurrentProcess(),modules,C.sizeof(modules),C.byref(needed)):raise C.WinError(C.get_last_error())
    if needed.value>C.sizeof(modules):raise RuntimeError('Loaded module inventory exceeds provenance capacity')
    result=[]
    for module in modules[:needed.value//C.sizeof(C.c_void_p)]:
        path=C.create_unicode_buffer(32768);length=kernel.GetModuleFileNameW(module,path,len(path))
        if not length or length>=len(path):raise C.WinError(C.get_last_error())
        file=Path(path.value).resolve();name=file.name.lower()
        if name.startswith('mkl_') or name.startswith('libiomp'):
            in_project=file.is_relative_to(ROOT)
            result.append({'path':str(file),'in_project':in_project,
                           'sha256':hashlib.sha256(file.read_bytes()).hexdigest() if in_project else None})
    return sorted(result,key=lambda item:item['path'].lower())

class CpuEvaluationError(ValueError):
    def __init__(self,message,objectives=None,diagnostics=None,codes=None):
        super().__init__(message);self.objectives=objectives;self.diagnostics=diagnostics;self.codes=codes

class CpuBackend:
    def __init__(self,workers=1,primary_max_s=6,fallback=True,dll=None,mkl=None):
        self.path=native_path(Path(dll or ROOT/'build/cpu-backend-v1-reviewed/kaks_cpu_backend.dll'))
        self.mkl_path=blas_runtime(ROOT,mkl)
        self.core=SharedCore(self.path);self.dll=self.core.dll;self.handle=C.c_void_p();self.pairs=[]
        self.buffer_capacity=0;self.buffer_objectives=self.buffer_diagnostics=self.buffer_codes=None
        self.pair_table=None;self.error=C.create_string_buffer(1024)
        self.directories=[dll_directory(str(self.mkl_path.parent))]
        d=self.dll;u=C.c_uint32;v=C.c_void_p;p=C.POINTER(C.c_double);z=C.c_uint64
        d.kaks_cpu_abi_version.argtypes=[];d.kaks_cpu_abi_version.restype=u
        d.kaks_cpu_workspace_doubles.argtypes=[];d.kaks_cpu_workspace_doubles.restype=z
        d.kaks_cpu_create.argtypes=[C.c_wchar_p,u,u,u,C.POINTER(v),C.c_char_p,u]
        d.kaks_cpu_free.argtypes=[v];d.kaks_cpu_free.restype=None
        d.kaks_cpu_runtime_info.argtypes=[v,C.c_wchar_p,u,C.c_char_p,u,C.POINTER(u),C.c_char_p,u]
        d.kaks_cpu_evaluate_batch.argtypes=[v,C.POINTER(v),u,C.POINTER(C.c_int64),p,z,p,p,C.POINTER(u),z,C.c_char_p,u]
        d.kaks_cpu_evaluate_one.argtypes=[v,v,u,p,p,z,p,p,p,C.c_char_p,u]
        d.kaks_cpu_evaluate_expanded.argtypes=[v,v,p,p,C.c_double,C.c_double,p,z,p,p,p,C.c_char_p,u]
        d.kaks_cpu_matrix_exp.argtypes=[v,p,p,p,z,p,C.c_char_p,u]
        for name in ('create','runtime_info','evaluate_batch','evaluate_one','evaluate_expanded','matrix_exp'):
            getattr(d,'kaks_cpu_'+name).restype=C.c_int
        if d.kaks_cpu_abi_version()!=1:raise RuntimeError('Unknown CPU objective ABI')
        self.workspace_doubles=int(d.kaks_cpu_workspace_doubles());self.workspace=(C.c_double*self.workspace_doubles)()
        error=C.create_string_buffer(1024)
        try:
            self._check(d.kaks_cpu_create(str(self.mkl_path),workers,primary_max_s,int(bool(fallback)),C.byref(self.handle),error,len(error)),error)
            path=C.create_unicode_buffer(32768);version=C.create_string_buffer(1024);actual_workers=u()
            self._check(d.kaks_cpu_runtime_info(self.handle,path,len(path),version,len(version),C.byref(actual_workers),error,len(error)),error)
            self.loaded_mkl_path=Path(path.value).resolve();self.mkl_version=version.value.decode('utf-8',errors='replace');self.workers=actual_workers.value
            if self.loaded_mkl_path!=self.mkl_path:raise RuntimeError('Actually loaded MKL path differs from requested runtime')
        except BaseException:
            self.close();raise

    @staticmethod
    def _check(code,error,**saved):
        if code:raise CpuEvaluationError(error.value.decode('utf-8',errors='replace'),**saved)
    def require_open(self):
        if not self.handle.value:raise RuntimeError('CPU objective session is closed')
    def prepare(self,pair_id,exported):
        self.require_open();pair=self.core.prepare(pair_id,exported);self.pairs.append(pair);self.pair_table=None;return pair
    def _pair_table(self):
        for pair in self.pairs:pair.require_open()
        if self.pair_table is None:self.pair_table=(C.c_void_p*len(self.pairs))(*(p.handle for p in self.pairs))
        return self.pair_table
    def objectives_buffered(self,slots,parameters,count=None,diagnostics=False):
        """Borrow contiguous ctypes input; return an exact-count ctypes view.

        Input objects remain alive for the synchronous native call. Output and
        raw diagnostics views are valid only until the next call on this
        provider. Do not modify/free pairs or buffers during a call. No Python
        per-row/list conversion occurs unless explicitly requested or an error
        must preserve failed rows for diagnosis. Native validation remains
        authoritative, including repeats, finite parameters, capacities.
        """
        self.require_open()
        if not isinstance(slots,C.Array) or slots._type_ is not C.c_int64:raise ValueError('Expected contiguous ctypes int64 slots')
        if not isinstance(parameters,C.Array) or parameters._type_ is not C.c_double:raise ValueError('Expected contiguous ctypes double parameters')
        count=len(slots) if count is None else count
        if not isinstance(count,int) or count<0 or count>len(slots) or count*7>len(parameters):raise ValueError('Invalid borrowed request count/capacity')
        if count>self.buffer_capacity:
            capacity=max(count,max(64,self.buffer_capacity*2))
            self.buffer_objectives=(C.c_double*capacity)();self.buffer_diagnostics=(C.c_double*(capacity*8))();self.buffer_codes=(C.c_uint32*capacity)()
            self.buffer_capacity=capacity
        if not count:
            self.last_diagnostics=[] if diagnostics else None;self.last_codes=[] if diagnostics else None
            return (C.c_double*0)()
        code=self.dll.kaks_cpu_evaluate_batch(self.handle,self._pair_table(),len(self.pairs),slots,parameters,count,
            self.buffer_objectives,self.buffer_diagnostics,self.buffer_codes,self.buffer_capacity,self.error,len(self.error))
        out=(C.c_double*count).from_buffer(self.buffer_objectives)
        raw_diag=(C.c_double*(count*8)).from_buffer(self.buffer_diagnostics);raw_codes=(C.c_uint32*count).from_buffer(self.buffer_codes)
        self.last_diagnostic_buffer=raw_diag;self.last_code_buffer=raw_codes;self.last_count=count
        if code==2:self._check(code,self.error,objectives=list(out),diagnostics=[list(raw_diag[i*8:i*8+8]) for i in range(count)],codes=list(raw_codes))
        self._check(code,self.error)
        self.last_diagnostics=[list(raw_diag[i*8:i*8+8]) for i in range(count)] if diagnostics else None
        self.last_codes=list(raw_codes) if diagnostics else None
        return out
    def objectives(self,slots,parameters):
        self.require_open();slots=list(slots);parameters=list(parameters);count=len(slots)
        if len(parameters)!=count or any(len(x)!=7 for x in parameters):raise ValueError('Expected one seven-double request per slot')
        indices=(C.c_int64*count)(*slots);x=(C.c_double*(count*7))(*(v for row in parameters for v in row))
        return list(self.objectives_buffered(indices,x,count,diagnostics=True))
    def evaluate(self,pair,model,x,transition=False):
        self.require_open();pair.require_open()
        if len(x)!=7:raise ValueError('Expected seven raw doubles')
        params=(C.c_double*7)(*x);out=C.c_double();diag=(C.c_double*8)();p=(C.c_double*3721)() if transition else None
        error=C.create_string_buffer(1024);code=self.dll.kaks_cpu_evaluate_one(self.handle,pair.handle,model,params,self.workspace,self.workspace_doubles,C.byref(out),diag,p,error,len(error))
        self._check(code,error);return out.value,list(diag),list(p) if p is not None else None
    def evaluate_expanded(self,pair,pi,rates,omega,codon_time,transition=False):
        self.require_open();pair.require_open()
        if len(pi)!=64 or len(rates)!=6:raise ValueError('Expected packed pi64 and expanded rates6')
        pi=(C.c_double*64)(*pi);rates=(C.c_double*6)(*rates);out=C.c_double();diag=(C.c_double*8)();p=(C.c_double*3721)() if transition else None
        error=C.create_string_buffer(1024);code=self.dll.kaks_cpu_evaluate_expanded(self.handle,pair.handle,pi,rates,omega,codon_time,self.workspace,self.workspace_doubles,C.byref(out),diag,p,error,len(error))
        self._check(code,error);return out.value,list(diag),list(p) if p is not None else None
    def matrix_exp(self,matrix):
        self.require_open()
        if len(matrix)!=3721:raise ValueError('Expected 61x61 row-major matrix')
        a=(C.c_double*3721)(*matrix);out=(C.c_double*3721)();diag=(C.c_double*8)();error=C.create_string_buffer(1024)
        code=self.dll.kaks_cpu_matrix_exp(self.handle,a,out,self.workspace,self.workspace_doubles,diag,error,len(error));self._check(code,error)
        return list(out),list(diag)
    def provenance(self):
        return {'DLL':str(self.path),'DLL_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'build':json.loads(build_record(self.path).read_text(encoding='utf-8-sig')),
                'MKL_actual_path':str(self.loaded_mkl_path),'MKL_runtime_sha256':hashlib.sha256(self.loaded_mkl_path.read_bytes()).hexdigest(),
                'MKL_loaded_implementation_modules':mkl_modules(),
                'MKL_version':self.mkl_version,'workers':self.workers,'MKL_threads_per_worker':1,
                'workspace_doubles_per_worker':self.workspace_doubles,'workspace_bytes_per_worker':self.workspace_doubles*8,
                'pair_count':len(self.pairs),'GPU_execution':False,'optimizer_called':False}
    def close(self):
        if self.handle.value:self.dll.kaks_cpu_free(self.handle);self.handle=C.c_void_p()
        for pair in self.pairs:pair.close()
        self.pairs=[];self.pair_table=None
        if hasattr(self,'core'):self.core.close()
        for directory in self.directories:directory.close()
        self.directories=[]
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
