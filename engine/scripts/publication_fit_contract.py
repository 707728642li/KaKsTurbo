"""Declared raw parameter boxes for coherent publication fits and reports."""
from __future__ import annotations
import math
NK=(0,0,1,1,2,2,2,2,3,3,4,4,5,5)

def raw_bounds(cleaned_nt,snp,model,policy='historic'):
    if not isinstance(model,int) or not 0<=model<14:raise ValueError('Unknown model')
    if not math.isfinite(cleaned_nt) or cleaned_nt<=0 or not math.isfinite(snp) or not 0<=snp<=cleaned_nt:raise ValueError('Invalid cleaned sequence metrics')
    if policy not in ('historic','fixed'):raise ValueError('Unknown time-bound policy')
    historical=3*snp/cleaned_nt if snp/cleaned_nt>1e-6 else 1e-6
    lower=historical if policy=='historic' else 1e-6;nk=NK[model]
    lo=[lower]+[.01]*nk+[.001];hi=[3.]+[30.]*nk+[50.]
    lo+=[.001]*(7-len(lo));hi+=[50.]*(7-len(hi))
    return lo,hi

def declared_bounds(row,cleaned_nt,snp,model):
    """Old records without a policy are explicitly historical, never fixed."""
    policy=row.get('time_bound_policy','historic');lo,hi=raw_bounds(cleaned_nt,snp,model,policy)
    if 'lower' in row or 'upper' in row:
        if row.get('lower')!=lo or row.get('upper')!=hi:raise ValueError('Saved raw bounds disagree with the declared fit contract')
    elif policy!='historic':raise ValueError('Fixed-time fits require explicit raw-box metadata')
    return policy,lo,hi
