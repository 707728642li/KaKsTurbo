"""Native CPU FP64 fused F/log-gradient provider; importing performs no work.

A persistent pool owns one reusable tape/arena per worker. Kind0 evaluates F
and writes zero gradients, kind1 also returns raw-F derivatives in log(raw x)
coordinates. The optimizer alone divides both F and gradient by codon count L.
Use only after fixed28 and independent high-precision derivative gates pass.
"""
from __future__ import annotations
from product_v1_platform import blas_runtime, build_record, dll_directory, native_path
import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from shared_core import SharedCore
from cpu_backend import mkl_modules

ROOT=Path(__file__).resolve().parents[1]
DIAGNOSTICS=('norm1','squarings','adaptive_fallback','gemm_count',
             'minimum_observed_joint_probability','active_row_sum_error',
             'minimum_transition_entry','MKL_threads','forward_gemm_count',
             'reverse_gemm_count','active_parameters','request_kind')

@dataclass(frozen=True)
class NativeGradientPoint:
    objective: float
    gradient_raw: tuple[float,...]
    gradient_log: tuple[float,...]
    diagnostics: dict
    transition: tuple[float,...] | None=None

class CpuGradientError(ValueError):
    def __init__(self,message,objectives=None,gradient_log=None,diagnostics=None,codes=None):
        super().__init__(message)
        self.objectives,self.gradient_log,self.diagnostics,self.codes=objectives,gradient_log,diagnostics,codes

class CpuGradient:
    def __init__(self,workers=1,primary_max_s=6,tape_max_s=64,fallback=True,dll=None,mkl=None):
        self.path=native_path(Path(dll or ROOT/'build/cpu-gradient-v2/kaks_cpu_gradient.dll'))
        self.mkl_path=blas_runtime(ROOT,mkl)
        self.handle=C.c_void_p();self.pairs=[];self.pair_table=None
        self.buffer_capacity=0;self.buffer_objectives=self.buffer_gradients=self.buffer_diagnostics=self.buffer_codes=None
        self.error=C.create_string_buffer(1024);self.directories=[]
        self.primary_max_s,self.tape_max_s,self.fallback=primary_max_s,tape_max_s,bool(fallback)
        self.directories.append(dll_directory(str(self.mkl_path.parent)))
        try:
            self.core=SharedCore(self.path);self.dll=self.core.dll
            d=self.dll;u=C.c_uint32;v=C.c_void_p;p=C.POINTER(C.c_double);z=C.c_uint64
            d.kaks_cpu_gradient_abi_version.argtypes=[];d.kaks_cpu_gradient_abi_version.restype=u
            d.kaks_cpu_gradient_workspace_doubles.argtypes=[u];d.kaks_cpu_gradient_workspace_doubles.restype=z
            d.kaks_cpu_gradient_create.argtypes=[C.c_wchar_p,u,u,u,u,C.POINTER(v),C.c_char_p,u]
            d.kaks_cpu_gradient_free.argtypes=[v];d.kaks_cpu_gradient_free.restype=None
            d.kaks_cpu_gradient_runtime_info.argtypes=[v,C.c_wchar_p,u,C.c_char_p,u,C.POINTER(u),C.POINTER(u),C.c_char_p,u]
            d.kaks_cpu_gradient_evaluate_batch.argtypes=[v,C.POINTER(v),u,C.POINTER(C.c_int64),p,C.POINTER(u),z,p,p,p,C.POINTER(u),z,C.c_char_p,u]
            d.kaks_cpu_gradient_evaluate_one.argtypes=[v,v,u,p,u,p,z,p,p,p,p,p,C.c_char_p,u]
            for name in ('create','runtime_info','evaluate_batch','evaluate_one'):
                getattr(d,'kaks_cpu_gradient_'+name).restype=C.c_int
            if d.kaks_cpu_gradient_abi_version()!=1:raise RuntimeError('Unknown native CPU gradient ABI')
            self._check(d.kaks_cpu_gradient_create(str(self.mkl_path),workers,primary_max_s,tape_max_s,int(bool(fallback)),C.byref(self.handle),self.error,len(self.error)))
            path=C.create_unicode_buffer(32768);version=C.create_string_buffer(1024);actual_workers=u();actual_tape=u()
            self._check(d.kaks_cpu_gradient_runtime_info(self.handle,path,len(path),version,len(version),C.byref(actual_workers),C.byref(actual_tape),self.error,len(self.error)))
            self.loaded_mkl_path=Path(path.value).resolve();self.mkl_version=version.value.decode('utf-8',errors='replace')
            self.workers,self.tape_max_s=actual_workers.value,actual_tape.value
            if self.loaded_mkl_path!=self.mkl_path:raise RuntimeError('Actually loaded MKL differs from requested runtime')
            self.workspace_doubles=int(d.kaks_cpu_gradient_workspace_doubles(self.tape_max_s))
            self.workspace=(C.c_double*self.workspace_doubles)()
        except BaseException:
            self.close();raise

    def _check(self,code,**saved):
        if code:raise CpuGradientError(self.error.value.decode('utf-8',errors='replace'),**saved)
    def require_open(self):
        if not self.handle.value:raise RuntimeError('Native CPU gradient session is closed')
    def prepare(self,pair_id,exported):
        self.require_open();pair=self.core.prepare(pair_id,exported)
        pair.codon_count=int(exported[0][2]);self.pairs.append(pair);self.pair_table=None;return pair
    def _pair_table(self):
        for pair in self.pairs:pair.require_open()
        if self.pair_table is None:self.pair_table=(C.c_void_p*len(self.pairs))(*(p.handle for p in self.pairs))
        return self.pair_table
    def mixed_buffered(self,slots,parameters,kinds,count=None,diagnostics=False):
        """Borrow ctypes inputs and return F[count],flat g_log[count*7] views.

        Every buffer and pair must remain live for the synchronous call. Output
        views remain valid until the next provider call. Native validation is
        authoritative. Successful requests preserve input row order/repeats.
        """
        self.require_open()
        if not isinstance(slots,C.Array) or slots._type_ is not C.c_int64:raise ValueError('Expected ctypes int64 slots')
        if not isinstance(parameters,C.Array) or parameters._type_ is not C.c_double:raise ValueError('Expected ctypes double parameters')
        if not isinstance(kinds,C.Array) or kinds._type_ is not C.c_uint32:raise ValueError('Expected ctypes uint32 kinds')
        count=len(slots) if count is None else count
        if not isinstance(count,int) or count<0 or count>len(slots) or count>len(kinds) or count*7>len(parameters):raise ValueError('Invalid borrowed request count/capacity')
        if count>self.buffer_capacity:
            capacity=max(count,max(64,self.buffer_capacity*2))
            self.buffer_objectives=(C.c_double*capacity)();self.buffer_gradients=(C.c_double*(capacity*7))()
            self.buffer_diagnostics=(C.c_double*(capacity*12))();self.buffer_codes=(C.c_uint32*capacity)();self.buffer_capacity=capacity
        if not count:
            self.last_diagnostics=[] if diagnostics else None;self.last_codes=[] if diagnostics else None
            return (C.c_double*0)(),(C.c_double*0)()
        code=self.dll.kaks_cpu_gradient_evaluate_batch(self.handle,self._pair_table(),len(self.pairs),slots,parameters,kinds,count,
            self.buffer_objectives,self.buffer_gradients,self.buffer_diagnostics,self.buffer_codes,self.buffer_capacity,self.error,len(self.error))
        f=(C.c_double*count).from_buffer(self.buffer_objectives);g=(C.c_double*(count*7)).from_buffer(self.buffer_gradients)
        diag=(C.c_double*(count*12)).from_buffer(self.buffer_diagnostics);codes=(C.c_uint32*count).from_buffer(self.buffer_codes)
        self.last_diagnostic_buffer,self.last_code_buffer,self.last_count=diag,codes,count
        if code==2:self._check(code,objectives=list(f),gradient_log=[list(g[i*7:i*7+7]) for i in range(count)],
                              diagnostics=[list(diag[i*12:i*12+12]) for i in range(count)],codes=list(codes))
        self._check(code)
        self.last_diagnostics=[list(diag[i*12:i*12+12]) for i in range(count)] if diagnostics else None
        self.last_codes=list(codes) if diagnostics else None
        return f,g
    def mixed(self,slots,parameters,kinds):
        slots,parameters,kinds=list(slots),list(parameters),list(kinds);count=len(slots)
        if len(parameters)!=count or len(kinds)!=count or any(len(x)!=7 for x in parameters):raise ValueError('Expected x7 and kind for each slot')
        f,g=self.mixed_buffered((C.c_int64*count)(*slots),(C.c_double*(count*7))(*(v for row in parameters for v in row)),
                                (C.c_uint32*count)(*kinds),diagnostics=True)
        return list(f),[list(g[i*7:i*7+7]) for i in range(count)]
    def evaluate(self,pair,model,x,kind=1,transition=False):
        self.require_open();pair.require_open()
        if pair.core is not self.core:raise ValueError('Single point pair belongs to a different core/provider')
        if len(x)!=7:raise ValueError('Expected seven raw doubles')
        params=(C.c_double*7)(*x);out=C.c_double();log=(C.c_double*7)();raw=(C.c_double*7)();diag=(C.c_double*12)()
        p=(C.c_double*3721)() if transition else None
        code=self.dll.kaks_cpu_gradient_evaluate_one(self.handle,pair.handle,model,params,kind,self.workspace,self.workspace_doubles,
            C.byref(out),log,raw,diag,p,self.error,len(self.error));self._check(code)
        values=dict(zip(DIAGNOSTICS,diag))
        values.update(model_contract='pure-rates-strict-positive-v1',gradient_scope='conditional pi; complete Q/mr quotient rule',
                      gradient_coordinate='raw F w.r.t. log(raw x); not divided by codon_count L; inactive zero',
                      codon_count=pair.codon_count,GPU_execution=False,optimizer_called=False)
        return NativeGradientPoint(out.value,tuple(raw),tuple(log),values,tuple(p) if p is not None else None)
    def provenance(self):
        return {'DLL':str(self.path),'DLL_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'build':json.loads(build_record(self.path).read_text(encoding='utf-8-sig')),
                'MKL_actual_path':str(self.loaded_mkl_path),'MKL_runtime_sha256':hashlib.sha256(self.loaded_mkl_path.read_bytes()).hexdigest(),
                'MKL_loaded_implementation_modules':mkl_modules(),'MKL_version':self.mkl_version,
                'workers':self.workers,'MKL_threads_per_worker':1,'primary_max_s':self.primary_max_s,
                'tape_max_s':self.tape_max_s,'adaptive_fallback':self.fallback,
                'workspace_doubles_per_worker':self.workspace_doubles,'workspace_bytes_per_worker':self.workspace_doubles*8,
                'pair_count':len(self.pairs),'GPU_execution':False,'optimizer_called':False}
    def close(self):
        if self.handle.value:self.dll.kaks_cpu_gradient_free(self.handle);self.handle=C.c_void_p()
        for pair in self.pairs:pair.close()
        self.pairs=[];self.pair_table=None
        if hasattr(self,'core'):self.core.close()
        for directory in self.directories:directory.close()
        self.directories=[]
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
