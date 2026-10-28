"""Fresh derivatives at returned points; diagnostics, not an acceptance change.

The optimizer's last gradient need not belong to its best returned point.
This module records a new evaluation of that point with the verified provider.
It does not infer global optimality, uncertainty, or identifiability.
"""
from __future__ import annotations
import ctypes as C
import math
from publication_fit_contract import NK

def projected_log_gradient(x,gradient_log,codon_count,lower,upper,active):
    if not codon_count>0 or not all(math.isfinite(v) for v in gradient_log):
        raise ValueError('Invalid fresh gradient or codon count')
    values=[gradient_log[i]/codon_count for i in range(active)]
    projected=[]
    for i,g in enumerate(values):
        at_lower=abs(x[i]-lower[i])<=4*math.ulp(lower[i])
        at_upper=abs(x[i]-upper[i])<=4*math.ulp(upper[i])
        projected.append(0. if (at_lower and g>0.) or (at_upper and g<0.) else g)
    return values,projected

def audit_returned_gradients(provider,jobs,results,capacity=16384):
    if len(jobs)!=len(results) or capacity<1:raise ValueError('Invalid returned-point audit batch')
    summary={'schema':'kaksforge.returned_gradient/1','evaluated_models':0,
             'objective_absolute_tolerance':1e-10,'objective_relative_tolerance':1e-12,
             'max_objective_difference':0.,'max_projected_normalized_log_gradient':0.,
             'projected_gradient_threshold':1e-8,'within_threshold_count':0,
             'scope':'Fresh conditional F and d(F/L)/dlog(x) at each returned point; reported independently of finite-coordinate acceptance. No global optimality or uncertainty certificate.'}
    for start in range(0,len(jobs),capacity):
        current=jobs[start:start+capacity];points=results[start:start+capacity];n=len(current)
        slots=(C.c_int64*n)(*(r['slot'] for r in current))
        x=(C.c_double*(7*n))(*(v for p in points for v in p[:7]))
        kinds=(C.c_uint32*n)(*[1]*n)
        f,g=provider.mixed_buffered(slots,x,kinds,count=n)
        for j,(row,point) in enumerate(zip(current,points)):
            fresh=float(f[j]);difference=abs(fresh-point[7]);tol=1e-10+1e-12*max(abs(fresh),abs(point[7]))
            if not math.isfinite(fresh) or difference>tol:raise RuntimeError('Returned-point F disagrees with the frozen primitive tolerance')
            active=NK[row['slot']%14]+2;raw=list(g[j*7:j*7+7])
            if any(raw[i]!=0. for i in range(active,7)):raise RuntimeError('Fresh gradient has a nonzero inactive derivative')
            normalized,projected=projected_log_gradient(point[:7],raw,row['cleaned_nt']/3.,row['lower'],row['upper'],active)
            norm=max(abs(v) for v in projected)
            row['returned_gradient_diagnostic']={'F':fresh,'F_difference':difference,
                'gradient_log_raw_F':raw,'normalized_active_gradient_log':normalized,
                'projected_normalized_active_gradient_log':projected,'max_projected_norm':norm,
                'within_1e_minus_8':norm<=1e-8,'acceptance_policy_changed':False}
            summary['evaluated_models']+=1;summary['within_threshold_count']+=int(norm<=1e-8)
            summary['max_objective_difference']=max(summary['max_objective_difference'],difference)
            summary['max_projected_normalized_log_gradient']=max(summary['max_projected_normalized_log_gradient'],norm)
    summary['all_within_projected_threshold']=summary['evaluated_models']==summary['within_threshold_count']
    return summary
