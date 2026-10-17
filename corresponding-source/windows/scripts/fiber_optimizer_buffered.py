"""Borrowed CPU tensor requests for the independent original queued DLL.

A batch is valid until the next poll or submit on its owning optimizer. This
separate API avoids materializing millions of Python parameter objects.
"""
import ctypes as C
from dataclasses import dataclass
from fiber_optimizer import FiberOptimizer, ROOT, I32, U32, U64, D64, PI, PD, PU64


@dataclass(frozen=True)
class TensorRequestBatch:
    owner: object
    generation: int
    slots: object
    slots32: object
    parameters: object
    count: int
    completed: int

    def __len__(self):return self.count


class BufferedFiberOptimizer(FiberOptimizer):
    def __init__(self,dll_path=None,**options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-queued/fiber_optimizer.dll',**options)
        import torch
        self.torch=torch;self._capacity=0;self._generation=0
        self._count,self._done=U32(),U32();self._error=C.create_string_buffer(2048)

    def poll(self,capacity=None):
        self._open();capacity=self.task_count if capacity is None else capacity
        if capacity<1:raise ValueError('Add pairs before polling')
        if capacity>self._capacity:
            self._meta=(I32*(capacity*6))();self._x=(D64*(capacity*7))()
            self._tickets=(U64*capacity)();self._capacity=capacity
        self._generation+=1
        self._check(self.dll.kaks_fiber_poll(self.context,capacity,self._meta,self._x,
            self._tickets,C.byref(self._count),C.byref(self._done),self._error,len(self._error)),self._error)
        n=self._count.value;t=self.torch
        if not n:return TensorRequestBatch(self,self._generation,None,None,None,0,self._done.value)
        meta=t.frombuffer(self._meta,dtype=t.int32,count=n*6).reshape(n,6)
        slots32=meta[:,0].contiguous()
        slots=slots32.to(dtype=t.int64)
        x=t.frombuffer(self._x,dtype=t.float64,count=n*7).reshape(n,7)
        return TensorRequestBatch(self,self._generation,slots,slots32,x,n,self._done.value)

    def _batch(self,batch):
        self._open()
        if not isinstance(batch,TensorRequestBatch) or batch.owner is not self or batch.generation!=self._generation:
            raise RuntimeError('Stale borrowed batch; poll or submit invalidated its storage')
        if not len(batch):raise ValueError('No pending borrowed requests')

    def submit(self,batch,values,*,value_kind='negative_log_likelihood',eigen_warnings=None):
        self._batch(batch);t=self.torch
        if not isinstance(values,t.Tensor):values=t.tensor(values,dtype=t.float64,device='cpu')
        if values.device.type!='cpu' or values.dtype!=t.float64 or values.shape!=(len(batch),) or not values.is_contiguous():
            raise ValueError('Submit requires one contiguous CPU FP64 objective per request')
        if value_kind=='lnL':values=-values
        elif value_kind!='negative_log_likelihood':raise ValueError('Unknown objective sign')
        warnings=None
        if eigen_warnings is not None:
            eigen_warnings=list(eigen_warnings)
            if len(eigen_warnings)!=len(batch):raise ValueError('Warning count differs')
            warnings=(I32*len(batch))(*eigen_warnings)
        self._check(self.dll.kaks_fiber_submit(self.context,len(batch),C.cast(batch.slots32.data_ptr(),PI),
            self._tickets,C.cast(values.data_ptr(),PD),warnings,self._error,len(self._error)),self._error)
        self._generation+=1

    def mock_evaluate(self,batch):
        self._batch(batch)
        values,warnings=(D64*len(batch))(),(I32*len(batch))()
        self._check(self.dll.kaks_fiber_mock_evaluate(self.context,len(batch),C.cast(batch.slots32.data_ptr(),PI),
            self._tickets,values,warnings,self._error,len(self._error)),self._error)
        return list(values),list(warnings)


from fiber_optimizer_stagnation import StagnationFiberOptimizer


class BufferedStagnationFiberOptimizer(BufferedFiberOptimizer,StagnationFiberOptimizer):
    """Same borrowed request API with the separately validated policy3 DLL."""
    def __init__(self,dll_path=None,**options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-stagnation-v4/fiber_optimizer.dll',**options)


from fiber_optimizer_stagnation_v5 import PolishedStagnationFiberOptimizer


class BufferedPolishedFiberOptimizer(BufferedFiberOptimizer,PolishedStagnationFiberOptimizer):
    def __init__(self,dll_path=None,**options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-stagnation-v5/fiber_optimizer.dll',**options)
