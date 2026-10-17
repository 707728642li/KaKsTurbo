"""Explicit opt-in binding for the independent research refinement DLL."""
import ctypes as C
from fiber_optimizer import FiberOptimizer, ROOT, U32, D64, PD

DIAGNOSTIC_NAMES=('trigger_bits_cap1_repeat2_box4','primary_Iround','face_probes',
    'accepted_faces','restarts','last_restart_Iround','extra_Iround_sum',
    'initial_F','final_F','F_gain','restart_rollbacks','last_restart_return',
    'primary_requests','primary_cache_hits','feasible_baseline_F',
    'projection_F_minus_primary','F_gain_vs_feasible')


class StableFiberOptimizer(FiberOptimizer):
    def __init__(self, dll_path=None, **options):
        super().__init__(dll_path or ROOT/'build/fiber-optimizer-stable-feasible/fiber_optimizer.dll', **options)
        d=self.dll
        d.kaks_fiber_stable_policy_version.argtypes=[]
        d.kaks_fiber_stable_policy_version.restype=U32
        d.kaks_fiber_stable_diagnostic_columns.argtypes=[]
        d.kaks_fiber_stable_diagnostic_columns.restype=U32
        if d.kaks_fiber_stable_policy_version()!=2 or d.kaks_fiber_stable_diagnostic_columns()!=len(DIAGNOSTIC_NAMES):
            self.close()
            raise RuntimeError('Unknown optional refinement policy')
        d.kaks_fiber_stable_diagnostics.argtypes=[C.c_void_p,U32,PD,C.c_char_p,U32]
        d.kaks_fiber_stable_diagnostics.restype=C.c_int

    def refinement_diagnostics(self):
        self._open()
        values=(D64*(self.task_count*len(DIAGNOSTIC_NAMES)))()
        error=C.create_string_buffer(2048)
        self._check(self.dll.kaks_fiber_stable_diagnostics(self.context,self.task_count,
                    values,error,len(error)),error)
        width=len(DIAGNOSTIC_NAMES)
        return [dict(zip(DIAGNOSTIC_NAMES,values[i*width:(i+1)*width])) for i in range(self.task_count)]

    stable_diagnostics=refinement_diagnostics
