"""Experimental fused F/log-gradient queue; no mathematical provider hidden."""
from __future__ import annotations
import ctypes as C
from batched_optimizer import BatchedOptimizer,RequestBatch,ROOT
class GradientOptimizer(BatchedOptimizer):
    def __init__(self,max_jobs,**options):
        options.setdefault('dll_path',ROOT/'build/gradient-optimizer-v1/kaks_optimizer.dll');super().__init__(max_jobs,**options)
        try:
            d=self.dll;u=C.c_uint32;ps=C.POINTER(C.c_int64);pt=C.POINTER(C.c_uint64);pd=C.POINTER(C.c_double)
            d.kaks_opt_gradient_abi_version.argtypes=[];d.kaks_opt_gradient_abi_version.restype=u
            if d.kaks_opt_gradient_abi_version()!=1:raise RuntimeError('Unknown fused-gradient ABI')
            d.kaks_opt_poll_kinds.argtypes=[C.c_void_p,u,ps,pt,pd,C.POINTER(u),C.POINTER(u),C.POINTER(u),C.c_char_p,u]
            d.kaks_opt_submit_mixed.argtypes=[C.c_void_p,u,ps,pt,pd,pd,C.c_char_p,u]
            d.kaks_opt_poll_kinds.restype=d.kaks_opt_submit_mixed.restype=C.c_int
        except BaseException:
            self.close();raise
    def poll(self,capacity=None):
        self.require_open();capacity=capacity or max(1,self.job_count*14)
        slots=(C.c_int64*capacity)();tickets=(C.c_uint64*capacity)();x=(C.c_double*(capacity*7))();kinds=(C.c_uint32*capacity)();count=C.c_uint32();done=C.c_uint32()
        self.check(self.dll.kaks_opt_poll_kinds(self.handle,capacity,slots,tickets,x,kinds,C.byref(count),C.byref(done),self.error,len(self.error)))
        batch=RequestBatch(self,slots,tickets,x,count.value,done.value);batch.kinds=kinds;return batch
    def submit_mixed(self,batch,values,gradient_log):
        self.require_open()
        if batch.owner is not self:raise ValueError('Batch belongs to another optimizer')
        if not isinstance(values,C.Array):
            values=list(values)
            if len(values)!=len(batch):raise ValueError('Expected one F per request')
            values=(C.c_double*len(batch))(*values)
        if values._type_ is not C.c_double or len(values)!=len(batch):raise ValueError('Expected F[count] doubles')
        if not isinstance(gradient_log,C.Array):
            gradient_log=[list(row) for row in gradient_log]
            if len(gradient_log)!=len(batch) or any(len(row)!=7 for row in gradient_log):raise ValueError('Expected gradient_log[count,7]')
            gradient_log=(C.c_double*(len(batch)*7))(*(v for row in gradient_log for v in row))
        if gradient_log._type_ is not C.c_double or len(gradient_log)!=len(batch)*7:raise ValueError('Expected gradient_log[count,7] doubles')
        self.check(self.dll.kaks_opt_submit_mixed(self.handle,len(batch),batch.slots,batch.tickets,values,gradient_log,self.error,len(self.error)))
