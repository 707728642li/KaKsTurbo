"""Revision5: all-model finite-coordinate quality and bounded polishing."""
import ctypes as C
from fiber_optimizer import ROOT, U32, PD, D64
from fiber_optimizer_stagnation_v4 import ProtectedStagnationFiberOptimizer

POLISH_METRIC_NAMES=('measured','objective_improvement_threshold','initial_coordinate_gain',
    'remaining_coordinate_gain','threshold_reached','coordinate_searches','expansion_evaluations',
    'golden_evaluations','coordinate_probe_evaluations','strictly_accepted_evaluations',
    'polish_rounds','round_budget_exceeded','winner_verification_rejects',
    'max_rounds','max_golden_steps','max_expansion_steps')


class PolishedStagnationFiberOptimizer(ProtectedStagnationFiberOptimizer):
    def __init__(self,dll_path=None,**options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-stagnation-v5/fiber_optimizer.dll',**options)
        try:
            columns=self.dll.kaks_fiber_polish_metric_columns;columns.argtypes=[];columns.restype=U32
            if columns()!=len(POLISH_METRIC_NAMES):raise RuntimeError('Unknown polish metric schema')
            function=self.dll.kaks_fiber_polish_metrics
            function.argtypes=[C.c_void_p,U32,PD,C.c_char_p,U32];function.restype=C.c_int
        except Exception:self.close();raise

    def polish_metrics(self):
        self._open();width=len(POLISH_METRIC_NAMES);values=(D64*(self.task_count*width))();error=C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_polish_metrics(self.context,self.task_count,values,error,len(error)),error)
        return [dict(zip(POLISH_METRIC_NAMES,values[i*width:(i+1)*width])) for i in range(self.task_count)]
