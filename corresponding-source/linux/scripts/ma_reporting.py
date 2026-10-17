"""Explicit robust reporting for pairs with zero observed substitutions.

Positive optimizer lower bounds are numerical constraints. They must not turn
an unidentifiable Ka/Ks ratio from identical retained codons into an estimate.
This rule uses native paired-masked input statistics, never reference results.
Raw fitted models and the unadjusted MA row are retained for inspection.
"""
import math
from benchmark_cpu import SCHEMA

UNINFORMATIVE_FIELDS=('Ka','Ks','Ka/Ks','P-Value(Fisher)','S-Substitutions',
    'N-Substitutions','Fold-S-Substitutions(0:2:4)',
    'Fold-N-Substitutions(0:2:4)','Divergence-Time')


def robust_row(row,observed_substitutions):
    parts=row.rstrip('\r\n').split('\t')
    if len(parts)!=len(SCHEMA):raise ValueError('MA reporting requires all 22 columns')
    observed=float(observed_substitutions)
    if not math.isfinite(observed) or observed<0 or not observed.is_integer():
        raise ValueError('Observed substitutions must be a finite nonnegative integer')
    info={'pair':parts[0],'uninformative_zero_substitutions':observed==0,
          'observed_substitutions_after_paired_masking':observed,
          'reporting_policy':'kakscuda.zero_substitution_NA/1',
          'reference_values_used':False,'changed_fields':[]}
    if observed==0:
        if float(parts[SCHEMA.index('Substitutions')])!=0:
            raise ValueError('Native observed count and MA substitution count disagree')
        for name in UNINFORMATIVE_FIELDS:
            index=SCHEMA.index(name)
            if parts[index]!='NA':info['changed_fields'].append({'field':name,'unadjusted':parts[index],'reported':'NA'})
            parts[index]='NA'
        info['reason']='No substitutions in retained paired codons; positive numerical lower bounds do not identify Ka/Ks.'
    return '\t'.join(parts)+'\n',info
