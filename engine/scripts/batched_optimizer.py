"""Independent compact C++ bounded BFGS queue; no Torch or device import."""
from __future__ import annotations
from product_v1_platform import dll_directory, native_path
import ctypes as C
import os
from pathlib import Path
import threading
ROOT=Path(__file__).resolve().parents[1]
RESULT_WIDTH=18
class BatchedOptimizer:
    def __init__(self,max_jobs,max_iterations=500,gradient_tolerance=1e-8,quality_tolerance=1e-6,dll_path=None):
        self.path=native_path(Path(dll_path or ROOT/'build/batched-optimizer-v1-reviewed/kaks_optimizer.dll'))
        self.dirs=[dll_directory(str(p)) for p in (self.path.parent,ROOT/'envs/cpu-reference/Library/bin')]
        self.dll=C.CDLL(str(self.path));d=self.dll;u=C.c_uint32;pd=C.POINTER(C.c_double);ps=C.POINTER(C.c_int64);pt=C.POINTER(C.c_uint64);common=[C.c_char_p,u]
        d.kaks_opt_abi_version.argtypes=[];d.kaks_opt_abi_version.restype=u
        if d.kaks_opt_abi_version()!=1:raise RuntimeError('Unknown optimizer ABI')
        d.kaks_opt_create.argtypes=[u,u,C.c_double,C.c_double,C.POINTER(C.c_void_p)]+common
        d.kaks_opt_add.argtypes=[C.c_void_p,C.c_int64,u,pd,pd,pd,C.c_double]+common
        d.kaks_opt_poll.argtypes=[C.c_void_p,u,ps,pt,pd,C.POINTER(u),C.POINTER(u)]+common
        d.kaks_opt_submit.argtypes=[C.c_void_p,u,ps,pt,pd]+common
        d.kaks_opt_results.argtypes=[C.c_void_p,u,pd]+common
        d.kaks_opt_free.argtypes=[C.c_void_p];d.kaks_opt_free.restype=None
        for name in ('create','add','poll','submit','results'):getattr(d,'kaks_opt_'+name).restype=C.c_int
        self.handle=C.c_void_p();self.error=C.create_string_buffer(2048);self.owner=threading.get_ident();self.job_count=0
        self.check(d.kaks_opt_create(max_jobs,max_iterations,gradient_tolerance,quality_tolerance,C.byref(self.handle),self.error,len(self.error)))
    def check(self,code):
        if code:raise ValueError(self.error.value.decode('utf-8',errors='replace'))
    def require_open(self):
        if not self.handle.value:raise RuntimeError('Optimizer closed')
        if threading.get_ident()!=self.owner:raise RuntimeError('Optimizer requires creating thread')
    def add(self,slot,dimensions,initial,lower,upper,scale=1.):
        self.require_open()
        if not all(len(v)==7 for v in (initial,lower,upper)):raise ValueError('Expected7 parameters/bounds')
        x,lo,hi=[(C.c_double*7)(*v) for v in (initial,lower,upper)]
        self.check(self.dll.kaks_opt_add(self.handle,slot,dimensions,x,lo,hi,scale,self.error,len(self.error)));self.job_count+=1
    def poll(self,capacity=None):
        self.require_open();capacity=capacity or max(1,self.job_count*14)
        slots=(C.c_int64*capacity)();tickets=(C.c_uint64*capacity)();x=(C.c_double*(capacity*7))();count=C.c_uint32();done=C.c_uint32()
        self.check(self.dll.kaks_opt_poll(self.handle,capacity,slots,tickets,x,C.byref(count),C.byref(done),self.error,len(self.error)))
        return RequestBatch(self,slots,tickets,x,count.value,done.value)
    def submit(self,batch,values):
        self.require_open()
        if batch.owner is not self:raise ValueError('Batch belongs to another optimizer')
        if isinstance(values,C.Array):
            if values._type_ is not C.c_double or len(values)!=batch.count:raise ValueError('Expected contiguous objective doubles')
            out=values
        else:
            values=list(values)
            if len(values)!=batch.count:raise ValueError('Expected one objective per ticket')
            out=(C.c_double*len(values))(*values)
        self.check(self.dll.kaks_opt_submit(self.handle,batch.count,batch.slots,batch.tickets,out,self.error,len(self.error)))
    def results(self):
        self.require_open();out=(C.c_double*(self.job_count*RESULT_WIDTH))()
        self.check(self.dll.kaks_opt_results(self.handle,self.job_count,out,self.error,len(self.error)))
        return [list(out[i*RESULT_WIDTH:(i+1)*RESULT_WIDTH]) for i in range(self.job_count)]
    def close(self):
        if self.handle.value:self.require_open();self.dll.kaks_opt_free(self.handle);self.handle=C.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
class RequestBatch:
    def __init__(self,owner,slots,tickets,parameters,count,completed):self.owner,self.slots,self.tickets,self.parameters,self.count,self.completed=owner,slots,tickets,parameters,count,completed
    def __len__(self):return self.count
    def rows(self):
        return [(self.slots[i],list(self.parameters[i*7:i*7+7])) for i in range(self.count)]
