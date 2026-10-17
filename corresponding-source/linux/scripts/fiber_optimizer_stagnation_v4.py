"""Independent protected-cache revision of policy3, with the same 23 diagnostics."""
import ctypes as C
from fiber_optimizer import ROOT, U32
from fiber_optimizer_stagnation import StagnationFiberOptimizer

PROTECTED_METRIC_NAMES=('feasible_protected_hits','best_protected_hits',
    'feasible_protected_updates','best_protected_updates')


class ProtectedStagnationFiberOptimizer(StagnationFiberOptimizer):
    def __init__(self,dll_path=None,**options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-stagnation-v4/fiber_optimizer.dll',**options)
        try:
            function=self.dll.kaks_fiber_protected_cache_metrics
            function.argtypes=[C.c_void_p,U32,C.POINTER(C.c_uint64),C.c_char_p,U32]
            function.restype=C.c_int
        except Exception:
            self.close()
            raise

    def protected_cache_metrics(self):
        self._open();width=len(PROTECTED_METRIC_NAMES)
        values=(C.c_uint64*(self.task_count*width))();error=C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_protected_cache_metrics(self.context,self.task_count,values,error,len(error)),error)
        return [dict(zip(PROTECTED_METRIC_NAMES,values[i*width:(i+1)*width])) for i in range(self.task_count)]
