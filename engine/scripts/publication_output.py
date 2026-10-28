"""Complete coherent MA output from the fourteen explicitly saved fitted slots.

The GPU prototype currently uses a one-thread CPU objective to rescore the
averaged state. This component is included in tool wall time and recorded.
Historical 22-column reproduction remains a separate estimator and mode.
"""
from __future__ import annotations
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
from publication_statistics import aggregate_fitted_pair
from shared_statistics import SharedStatistics

FIELDS=('Sequence','Method','Ka','Ks','Ka/Ks','S-sites','N-sites',
        'Expected-S-events','Expected-N-events','Observed-SNP','Codon-path-time',
        'Divergence-time','lnL','Status','Best-model','Iteration-capped-models')
MODELS=('JC','F81','K2P','HKY','TNEF','TN','K3P','K3PUF','TIMEF','TIM','TVMEF','TVM','SYM','GTR')


def scalar(value):
    if value is None:return 'NA'
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Nonfinite report scalar')
        return format(value,'.17g')
    return str(value)


def write_complete_ma(prepared,model_rows,directory,backend,provider):
    """Write ordered MA rows and all candidate/statistical diagnostics.

    A quality-failed candidate is retained and makes MA unavailable. No model
    subset or p-value is fabricated. Return execution-scope metadata.
    """
    directory=Path(directory);table=[];summaries=[]
    if len(model_rows)!=len(prepared.ids)*14:raise ValueError('Incomplete candidate records')
    byslot={row['slot']:row for row in model_rows}
    if set(byslot)!=set(range(len(prepared.ids)*14)):raise ValueError('Duplicate or missing candidate slots')
    with ExitStack() as stack:
        statistics=SharedStatistics()
        for directory_handle in statistics.directories:stack.callback(directory_handle.close)
        reuse_fit_provider=backend=='cpu' and hasattr(provider,'evaluate_expanded')
        if reuse_fit_provider:ma_provider=provider
        else:
            from cpu_backend import CpuBackend
            ma_provider=stack.enter_context(CpuBackend(workers=1))
        for i,name in enumerate(prepared.ids):
            exported=(prepared.meta[i],prepared.values[i],prepared.z0[i],prepared.z1[i],prepared.weights[i],prepared.gc[i])
            stat_pair=stack.enter_context(statistics.prepare(name,exported))
            cpu_pair=ma_provider.pairs[i] if reuse_fit_provider else ma_provider.prepare(name,exported)
            def score(state):
                return ma_provider.evaluate_expanded(cpu_pair,state.pi,state.rates,state.values[4],state.x[0])[0]
            rows=[byslot[i*14+m] for m in range(14)]
            summary=aggregate_fitted_pair(statistics,stat_pair,rows,score,expected_pair_id=name)
            summary['fitting_provenance']='models.json; common bounded optimizer, fourteen explicit slots'
            summary['iteration_capped_models']=[r['model'] for r in rows if r.get('iteration_limit_reached',False)]
            summary['MA_score_backend']='shared-core CPU FP64 T18/adaptive; one worker for GPU runs'
            summaries.append(summary)
            ma=summary['MA'];values=ma.get('averaged',{}).get('scientific_report',{})
            best=ma.get('best_model')
            table.append(dict(zip(FIELDS,(name,'MA',values.get('Ka'),values.get('Ks'),values.get('Ka_Ks'),
                values.get('S_sites'),values.get('N_sites'),values.get('expected_S_events'),values.get('expected_N_events'),
                summary['masked_observed_SNP'],values.get('x0'),values.get('divergence_time'),values.get('lnL'),
                ma['status'],None if best is None else MODELS[best],','.join(summary['iteration_capped_models']) or 'none'))))
        # CPU provenance is recorded even when the main fit runs on the GPU.
        ma_provenance=ma_provider.provenance()
    (directory/'pair_statistics.json').write_text(json.dumps(summaries,indent=None,allow_nan=False)+'\n',encoding='utf-8')
    with (directory/'result.tsv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t',lineterminator='\n');writer.writerow(FIELDS)
        writer.writerows([[scalar(row[field]) for field in FIELDS] for row in table])
    counts={status:sum(row['Status']==status for row in table) for status in ('available','unidentifiable','unavailable')}
    return {'schema':'kakscuda.complete_coherent_MA_output/1','row_count':len(table),'columns':list(FIELDS),
        'MA_status_counts':counts,'candidate_slot_count':len(model_rows),
        'iteration_capped_model_count':sum(len(s['iteration_capped_models']) for s in summaries),
        'MA_score_backend':'CPU','MA_score_backend_provenance':ma_provenance,
        'GPU_fit_includes_CPU_final_MA_rescoring':backend=='gpu',
        'MA_reuses_fit_provider':reuse_fit_provider,
        'interpretation':'Conditional plug-in AICc and parameter averaging; finite coordinate probes do not certify KKT or global optimality. No uncertainty/Fisher significance test.'}
