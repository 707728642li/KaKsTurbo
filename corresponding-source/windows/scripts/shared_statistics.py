"""Fixed-parameter coherent statistics / conditional MA, standard library only.

Caller-supplied lnL is not evaluated here. These results are provisional fixed
parameter calculations, never a fitting/optimization success certificate.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from shared_core import SharedCore, STRICT_POSITIVE, WIDTH

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['lnL','AICc','x0','divergence_time','omega','rs','ra0','ra','mr','S_sites','N_sites',
        'Ks','Ka','Ka_Ks','expected_S_events','expected_N_events',
        'observed_rescaled_S_substitutions','observed_rescaled_N_substitutions',
        'codon_count','observed_SNP','parameter_count']
FLAGS={1:'zero_observed_unidentifiable',2:'zero_scale',4:'no_synonymous_flow',8:'no_nonsynonymous_flow',
       16:'short_AICc_undefined',32:'score_unavailable',64:'MA_AICc_undefined',128:'outside_source_fit_box',
       256:'fixed_parameter_only',512:'zero_time'}
MA_FLAGS={1:'incomplete_models',2:'unavailable_candidate_criterion',4:'MA_score_not_yet_evaluated',
          8:'conditional_rate_defaulted',16:'fixed_parameter_only',32:'averaged_process_unavailable'}
STATUS=['available','unidentifiable','unavailable']


class ModelStatistics(C.Structure):
    _fields_=[('model',C.c_uint32),('support_policy',C.c_uint32),('status',C.c_uint32),('flags',C.c_uint32),
              ('pair_fingerprint',C.c_uint64),('math_mask',C.c_uint64),('report_mask',C.c_uint64),
              ('values',C.c_double*21),('rates',C.c_double*6),('pi',C.c_double*64),('x',C.c_double*7)]
    def as_dict(self):
        import math
        clean=lambda values:[v if math.isfinite(v) else None for v in values]
        return {'model':self.model,'support_policy':self.support_policy,'status':STATUS[self.status],
                'flags':[label for flag,label in FLAGS.items() if self.flags&flag],
                'pair_fingerprint_hex':f'{self.pair_fingerprint:016x}',
                'mathematical':{name:float(self.values[i]) if self.math_mask&(1<<i) else None for i,name in enumerate(FIELDS)},
                'scientific_report':{name:float(self.values[i]) if self.report_mask&(1<<i) else None for i,name in enumerate(FIELDS)},
                'rates':clean(self.rates),'pi':clean(self.pi),'x':clean(self.x),
                'provisional_fixed_parameters':True,'fitting_performed':False}


class MaStatistics(C.Structure):
    _fields_=[('status',C.c_uint32),('flags',C.c_uint32),('present_mask',C.c_uint32),('unavailable_mask',C.c_uint32),
              ('best_model',C.c_int32),('rate_default_mask',C.c_uint32),('weights',C.c_double*14),
              ('rate_weight_masses',C.c_double*5),('averaged',ModelStatistics)]
    def as_dict(self):
        import math
        clean=lambda values:[v if math.isfinite(v) else None for v in values]
        return {'status':STATUS[self.status],'flags':[label for flag,label in MA_FLAGS.items() if self.flags&flag],
                'present_mask':self.present_mask,'unavailable_mask':self.unavailable_mask,
                'best_model':None if self.best_model<0 else self.best_model,
                'conditional_rate_default_mask':self.rate_default_mask,
                'weights':clean(self.weights),'conditional_rate_weight_masses':clean(self.rate_weight_masses),
                'averaged':self.averaged.as_dict(),'provisional_fixed_parameters':True,'fitting_performed':False}


class SharedStatistics(SharedCore):
    def __init__(self,path=None):
        super().__init__(path or ROOT/'build/shared-statistics-v1-checked/kaks_statistics.dll')
        d=self.dll;u=C.c_uint32;v=C.c_void_p;pd=C.POINTER(C.c_double);ps=C.POINTER(ModelStatistics);pm=C.POINTER(MaStatistics)
        for name in ('abi_version','model_size','ma_size'):
            getattr(d,'kaks_statistics_'+name).argtypes=[];getattr(d,'kaks_statistics_'+name).restype=u
        if d.kaks_statistics_abi_version()!=1:raise RuntimeError('Unknown statistics ABI')
        if d.kaks_statistics_model_size()!=C.sizeof(ModelStatistics) or d.kaks_statistics_ma_size()!=C.sizeof(MaStatistics):
            raise RuntimeError('Statistics C ABI layout does not match ctypes')
        d.kaks_statistics_fixed.argtypes=[v,u,u,pd,u,C.c_double,ps,C.c_char_p,u]
        d.kaks_statistics_average.argtypes=[v,ps,u,pm,C.c_char_p,u]
        d.kaks_statistics_ma_q.argtypes=[v,pm,pd,u,C.c_char_p,u]
        d.kaks_statistics_ma_score.argtypes=[v,pm,C.c_double,C.c_char_p,u]
        for name in ('fixed','average','ma_q','ma_score'):getattr(d,'kaks_statistics_'+name).restype=C.c_int
    def fixed(self,pair,model,x,lnL=None,support_policy=STRICT_POSITIVE):
        pair.require_open()
        if pair.core is not self:raise ValueError('Pair belongs to a different statistics binding')
        if len(x)!=7:raise ValueError('Seven raw parameter slots required')
        result=ModelStatistics();error=C.create_string_buffer(1024)
        code=self.dll.kaks_statistics_fixed(pair.handle,model,support_policy,(C.c_double*7)(*x),
            int(lnL is not None),0. if lnL is None else lnL,C.byref(result),error,len(error))
        self.check(code,error);return result
    def average(self,pair,models):
        pair.require_open()
        if pair.core is not self:raise ValueError('Pair belongs to a different statistics binding')
        values=(ModelStatistics*len(models))(*models);out=MaStatistics();error=C.create_string_buffer(1024)
        code=self.dll.kaks_statistics_average(pair.handle,values,len(models),C.byref(out),error,len(error))
        self.check(code,error);return out
    def ma_q(self,pair,averaged):
        pair.require_open();out=(C.c_double*WIDTH)();error=C.create_string_buffer(1024)
        if pair.core is not self:raise ValueError('Pair belongs to a different statistics binding')
        code=self.dll.kaks_statistics_ma_q(pair.handle,C.byref(averaged),out,WIDTH,error,len(error))
        self.check(code,error);return out
    def attach_ma_lnL(self,pair,averaged,lnL):
        pair.require_open();error=C.create_string_buffer(1024)
        if pair.core is not self:raise ValueError('Pair belongs to a different statistics binding')
        code=self.dll.kaks_statistics_ma_score(pair.handle,C.byref(averaged),lnL,error,len(error))
        self.check(code,error);return averaged
