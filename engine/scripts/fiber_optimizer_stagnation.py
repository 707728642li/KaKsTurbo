"""Explicit experimental policy3 binding; no CUDA import/execution."""
import ctypes as C
from fiber_optimizer import FiberOptimizer, ROOT, U32, D64, PD
from fiber_optimizer_stable import DIAGNOSTIC_NAMES as BASE_NAMES

DIAGNOSTIC_NAMES=BASE_NAMES+('stagnation_exit_count','primary_stagnation_Iround',
    'last_stagnation_Iround','stagnation_threshold','final_coordinate_probe_gain','final_coordinate_no_decrease')


class StagnationFiberOptimizer(FiberOptimizer):
    def __init__(self,dll_path=None,**options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-stagnation-v3/fiber_optimizer.dll',**options)
        d=self.dll
        d.kaks_fiber_stable_policy_version.argtypes=[];d.kaks_fiber_stable_policy_version.restype=U32
        d.kaks_fiber_stable_diagnostic_columns.argtypes=[];d.kaks_fiber_stable_diagnostic_columns.restype=U32
        if d.kaks_fiber_stable_policy_version()!=3 or d.kaks_fiber_stable_diagnostic_columns()!=len(DIAGNOSTIC_NAMES):
            self.close();raise RuntimeError('Unknown experimental stagnation policy')
        d.kaks_fiber_stable_diagnostics.argtypes=[C.c_void_p,U32,PD,C.c_char_p,U32]
        d.kaks_fiber_stable_diagnostics.restype=C.c_int

    def refinement_diagnostics(self):
        self._open();width=len(DIAGNOSTIC_NAMES)
        values=(D64*(self.task_count*width))();error=C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_stable_diagnostics(self.context,self.task_count,values,error,len(error)),error)
        return [dict(zip(DIAGNOSTIC_NAMES,values[i*width:(i+1)*width])) for i in range(self.task_count)]

    stable_diagnostics=refinement_diagnostics
