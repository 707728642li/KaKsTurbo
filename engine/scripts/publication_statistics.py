"""Bridge declared common-optimizer results to the coherent statistics module.

All fourteen slots remain present. Unavailable/failed fits cannot silently become
a smaller model average. This helper neither fits models nor imports GPU code.
"""
from __future__ import annotations
import math
from check_publication_simulations import MODELS
from shared_core import STRICT_POSITIVE
from publication_fit_contract import declared_bounds

NK=(0,0,1,1,2,2,2,2,3,3,4,4,5,5)
QUALITY_TOLERANCE=1e-6


def aggregate_fitted_pair(statistics, prepared_pair, model_rows, ma_objective=None, expected_pair_id=None):
    """Return a JSON-safe complete 14-slot summary for one prepared pair.

    statistics: SharedStatistics instance owning prepared_pair.
    model_rows: exactly the corresponding fourteen benchmark models.json rows.
    expected_pair_id: the caller's prepared ID; supply this to validate the fit
      rows. The frozen PairData export contains numeric inputs, not its ID.
    ma_objective: optional callable(KaksModelStatistics) -> positive F for the
      averaged immutable state (pi64, rates6, omega, x0), using the SAME model
      contract. It is called once only after all candidate gates/criteria pass.
      A CPU example is in docs/coherent_statistics_contract.md.

    Scores are read from result18[7] and converted to lnL=-F. Status1 plus a
    finite coordinate probe is a local finite-test gate, not convergence or a
    global optimum proof. No fitted-model uncertainty or Fisher P value exists
    in this prototype. Caller must record input/model/backend provenance.
    """
    prepared_pair.require_open()
    meta,values,_,_,_,_=prepared_pair.export();L=int(meta[2]);snp=values[1]
    if len(model_rows)!=14:raise ValueError('Exactly fourteen model records required, including unavailable slots')
    ordered=[None]*14;pair_label=None;pair_index=None
    for row in model_rows:
        slot=row.get('slot')
        if not isinstance(slot,int) or slot<0:raise ValueError('A nonnegative integer slot is required')
        model=slot%14
        if row.get('model')!=MODELS[model]:raise ValueError('Model label and slot disagree')
        if not isinstance(row.get('pair'),str) or not row['pair']:raise ValueError('Pair identifier is required')
        if expected_pair_id is not None and row['pair']!=expected_pair_id:raise ValueError('Unexpected fit pair identifier')
        if ordered[model] is not None:raise ValueError('Duplicate model record')
        if pair_label is not None and (row.get('pair')!=pair_label or slot//14!=pair_index):raise ValueError('Mixed pair records')
        pair_label=row.get('pair');pair_index=slot//14;ordered[model]=row
        if row.get('cleaned_nt')!=values[0] or row.get('observed_substitutions')!=snp:raise ValueError('Fit and prepared input statistics disagree')
    if any(row is None for row in ordered):raise ValueError('Missing model record')
    policies={row.get('time_bound_policy','historic') for row in ordered}
    if len(policies)!=1:raise ValueError('All fourteen candidates require the same declared time-bound policy')
    candidates=[];summaries=[];rejected=[];unavailable=[]
    for model,row in enumerate(ordered):
        result=row.get('result18');entry={'slot':row['slot'],'model':MODELS[model],
            'optimizer_status':row.get('status'),'optimizer_executed':bool(row.get('optimizer_executed')),
            'finite_coordinate_gate':False,'criterion_available':False,'statistics':None}
        if result is None:
            unavailable.append(model);entry['availability']='unidentifiable' if snp==0 else 'unavailable'
            entry['reason']='No model point or objective exists; no finite criterion is fabricated'
            summaries.append(entry);continue
        if not isinstance(result,(list,tuple)) or len(result)!=18 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in result):
            rejected.append(model);entry.update(availability='unavailable',reason='Result18 contains missing/nonfinite fields');summaries.append(entry);continue
        np=NK[model]+2;policy,lo,hi=declared_bounds(row,values[0],snp,model)
        entry['time_bound_policy']=policy
        inside=all(lo[j]<=result[j]<=hi[j] for j in range(np))
        reported_gap=result[17];actual_gap=result[7]-result[16]
        gap_roundoff=4*max(math.ulp(result[7]),math.ulp(result[16]),math.ulp(reported_gap))
        consistent_best=(0.<=result[16]<=result[7] and result[16]<=result[8] and
                         abs(reported_gap-actual_gap)<=gap_roundoff)
        quality=(bool(row.get('optimizer_executed')) and result[14]==1. and inside and
                 0.<=result[10]<=QUALITY_TOLERANCE and 0.<=result[17]<=QUALITY_TOLERANCE and
                 result[7]>=0. and result[7]<=result[8] and consistent_best)
        if 'quality_passed' in row:quality=quality and row['quality_passed'] is True
        entry['finite_coordinate_gate']=quality
        try:
            stat=statistics.fixed(prepared_pair,model,result[:7],lnL=-result[7],support_policy=STRICT_POSITIVE)
            entry['statistics']=stat.as_dict();entry['criterion_available']=bool(stat.math_mask&2)
            entry['availability']=entry['statistics']['status'];candidates.append(stat)
        except (ValueError,RuntimeError) as error:
            quality=False;entry['finite_coordinate_gate']=False;entry['availability']='unavailable';entry['reason']=str(error)
        if not quality:rejected.append(model)
        summaries.append(entry)
    summary={'schema':'kakscuda.coherent_fitted_pair_statistics/1','pair':pair_label,'pair_index':pair_index,
        'cleaned_nt_length':int(values[0]),'codon_count':L,'masked_observed_SNP':snp,
        'model_contract':'pure-rates-strict-positive-v1','models':summaries,
        'time_bound_policy':next(iter(policies)),
        'model_coverage':14,'unavailable_models':unavailable,'quality_failed_models':rejected,
        'quality_scope':'Bounded finite coordinate probes <=1e-6; no returned-point KKT/global-optimum certificate',
        'statistics_scope':'Coherent sites/path dS/dN and conditional plug-in AICc; no Fisher P value or full legacy22 report'}
    if unavailable or rejected or len(candidates)!=14:
        summary['MA']={'status':'unidentifiable' if snp==0 else 'unavailable','weights':None,'lnL':None,'AICc':None,
            'reason':'All fourteen model points, valid criteria, and finite-fit gates are required; no subset averaging',
            'model_count':14,'unavailable_models':unavailable,'quality_failed_models':rejected}
        return summary
    average=statistics.average(prepared_pair,candidates)
    if not average.unavailable_mask and average.present_mask==0x3fff and average.best_model>=0:
        if ma_objective is not None:
            F=ma_objective(average.averaged)
            if not isinstance(F,(int,float)) or not math.isfinite(F) or F<0.:
                raise ValueError('MA evaluator must return finite nonnegative negative-log-likelihood F')
            statistics.attach_ma_lnL(prepared_pair,average,-F)
    summary['MA']=average.as_dict()
    summary['MA']['score_evaluated_at_averaged_state']=bool(average.averaged.math_mask&1)
    summary['MA']['fitted_input_gate']='all fourteen explicit finite-coordinate-fit gates validated'
    # The native statistics flag states no fitting was performed inside that
    # module; fitting provenance belongs to the caller/driver, never this hook.
    summary['MA']['native_statistics_module_performed_fitting']=False
    return summary
