"""Coherent CPU/GPU fitting with fourteen candidates and explicit diagnostics.

Isolated observed-output plus packed observed-adjoint and structural Jacobian VJP; full upstream matrices and active-free-subspace BFGS candidate fit loop.
The outer process harness supplies the complete process timing scope.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes as C
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from batched_optimizer import BatchedOptimizer
from benchmark_cpu import ROOT,dump
from kakscuda import PreparedAxtPairs,parse_input,select_device
from product_v1_engine_runtime import process_affinity
from product_v1_platform import native_path
from publication_provenance import source_snapshot,binary_snapshot,sha,verify_source_snapshot
from torch_matrix_prototype import MODELS,NKAPPA
from publication_fit_contract import raw_bounds

def run(a):
    if a.derivatives!='analytic' or a.time_bound!='fixed' or a.frequency_truth or a.frequency_scenarios:raise ValueError('Product 1.0 retains the validated analytic, fixed-bound empirical-frequency contract')
    a.input=a.input.resolve();a.output=a.output.resolve()
    if a.output.exists():raise ValueError('Do not overwrite an existing run')
    derivatives=getattr(a,'derivatives','fd')
    time_bound=getattr(a,'time_bound','historic');initial_time=getattr(a,'initial_time',.05)
    if time_bound not in ('historic','fixed') or not 1e-6<=initial_time<=3.:raise ValueError('Invalid time bound/initial time')
    frequency_truth=getattr(a,'frequency_truth',None);frequency_scenarios=getattr(a,'frequency_scenarios',None)
    if bool(frequency_truth)!=bool(frequency_scenarios):raise ValueError('Conditional true-pi fits require both truth and scenarios')
    if derivatives not in ('fd','analytic'):raise ValueError('Unknown derivative provider')
    default_optimizer='gradient-optimizer-schur-v1' if derivatives=='analytic' else 'batched-optimizer-v1-reviewed'
    optimizer_path=Path(getattr(a,'optimizer_dll',None) or ROOT/('build/'+default_optimizer+'/kaks_optimizer.dll')).resolve()
    optimizer_path=native_path(optimizer_path)
    from product_v1_engine_runtime import preflight_configuration
    validation_ticket,science_configuration,execution_configuration=preflight_configuration(a,optimizer_path)
    if a.backend=='gpu':execution_configuration.update(workers=1,runtime_engine='CUDA Driver API; single context exact observed-output plus packed observed-adjoint and structural587 VJP; cold-result reuse; no Torch import; no CUDA timing events',GPU_stream_count=a.stream_count,GPU_schedule=a.schedule,GPU_cold_reuse=True,GPU_event_instrumentation=False,GPU_reduced_candidate=('observed_output_shared_plus_structural587' if a.observed_layout=='shared' else 'observed_output_packed_plus_structural587'),GPU_selected_vjp_candidate='structural587',GPU_observed_layout=a.observed_layout,GPU_observed_output_installed=True)
    a.output.mkdir(parents=True);timings={};seeds=[Path(__file__).name,('publication_gpu_reduced_output_round6.py' if derivatives=='analytic' else 'publication_gpu_likelihood.py') if a.backend=='gpu' else ('cpu_gradient.py' if derivatives=='analytic' else 'cpu_backend.py')]
    if derivatives=='analytic':seeds.append('gradient_optimizer.py')
    if frequency_truth:seeds.append('publication_frequency_inputs_v2.py')
    if getattr(a,'entrypoint',None):seeds.append(a.entrypoint)
    if a.complete_ma:seeds.append('publication_output.py')
    closure=source_snapshot(a.output/'source_snapshot',seeds)
    gpu_inventory=None;monitor=None;device=None
    if a.backend=='gpu':
        from gpu_telemetry import GPUTelemetry,telemetry_uuid
        device,selected,gpu_inventory=select_device(a,a.output)
        if validation_ticket.get('publication_validation_authorized') and selected['uuid']!=execution_configuration['device_uuid']:
            raise ValueError('Selected physical GPU differs from the frozen execution profile')
        # Use a physical UUID before importing Torch; no ordinal ambiguity.
        import os
        os.environ['CUDA_VISIBLE_DEVICES']=selected['uuid'];device='cuda:0'
        monitor=GPUTelemetry(a.output,telemetry_uuid(device),500)
    backend_path=ROOT/('build/publication-graph-host-v1/torch_graph_host.dll' if a.backend=='gpu' else ('build/cpu-gradient-v2/kaks_cpu_gradient.dll' if derivatives=='analytic' else 'build/cpu-backend-v1-reviewed/kaks_cpu_backend.dll'))
    backend_path=native_path(backend_path)
    binary_sources=a.output/'binary_sources';binary_sources.mkdir()
    binaries={'optimizer':binary_snapshot(binary_sources,optimizer_path,'optimizer'),
        'backend':binary_snapshot(binary_sources,backend_path,'backend'),
        'prepare_adapter':binary_snapshot(binary_sources,ROOT/'build/native-adapter-checked/kaks_adapter.dll','prepare_adapter')}
    if a.complete_ma:
        binaries['statistics']=binary_snapshot(binary_sources,ROOT/'build/shared-statistics-v1-checked/kaks_statistics.dll','statistics')
        if a.backend=='gpu' or derivatives=='analytic':binaries['MA_rescoring_CPU']=binary_snapshot(binary_sources,ROOT/'build/cpu-backend-v1-reviewed/kaks_cpu_backend.dll','MA_rescoring_CPU')
    rows=[];wave_count=0;requests=0;started=time.perf_counter();deadline=started+a.max_seconds
    try:
        with process_affinity() as topo,ExitStack() as stack:
            data=a.input.read_bytes();records=parse_input(data)
            if a.count is None:a.count=len(records)
            if not 1<=a.count<=1000:raise ValueError('Product engine accepts chunks of 1..1000 pairs')
            records=records[:a.count]
            if len(records)!=a.count:raise ValueError('Requested count exceeds input')
            prepared=stack.enter_context(PreparedAxtPairs(records,data));frequency_record={'policy':'Historical conditional empirical/F3x4 plug-in; fixed during optimization'}
            if frequency_truth:
                from publication_frequency_inputs_v2 import inject_truth_frequencies
                frequency_record=inject_truth_frequencies(prepared,frequency_truth,frequency_scenarios,allow_validation=validation_ticket['publication_validation_authorized'])
            timings['read_parse_prepare']=time.perf_counter()-started
            tick=time.perf_counter()
            if a.backend=='cpu':
                if derivatives=='analytic':
                    from cpu_gradient import CpuGradient
                    provider=stack.enter_context(CpuGradient(workers=a.workers,dll=backend_path))
                else:
                    from cpu_backend import CpuBackend
                    provider=stack.enter_context(CpuBackend(workers=a.workers,dll=backend_path))
                for i,name in enumerate(prepared.ids):provider.prepare(name,(prepared.meta[i],prepared.values[i],prepared.z0[i],prepared.z1[i],prepared.weights[i],prepared.gc[i]))
            else:
                if derivatives!='analytic':raise ValueError('Observed + structural VJP round6 requires the unchanged analytic fit contract')
                from publication_gpu_reduced_output_round6 import PublicationGPUGradient
                provider=PublicationGPUGradient(prepared,device,device_uuid=selected['uuid'],batch_tasks=a.batch_tasks,max_cached_buckets=getattr(a,'gpu_cache_buckets',128),graph_min_rows=32,profile_execution=getattr(a,'profile_execution',False),stream_count=a.stream_count,schedule=a.schedule,cold_reuse=True,observed_layout=a.observed_layout)
                stack.callback(provider.close)
                if provider.device_uuid!=monitor.device_uuid:raise ValueError('CUDA Driver/NVML UUID mismatch')
            timings['backend_setup']=time.perf_counter()-tick
            if derivatives=='analytic':
                from gradient_optimizer import GradientOptimizer
                optimizer=stack.enter_context(GradientOptimizer(len(records)*14,dll_path=optimizer_path,max_iterations=a.max_iterations))
            else:optimizer=stack.enter_context(BatchedOptimizer(len(records)*14,dll_path=optimizer_path,max_iterations=a.max_iterations))
            jobs=[]
            for pair,name in enumerate(prepared.ids):
                nt,snp=prepared.values[pair][:2];historic_lower=3*snp/nt if snp/nt>1e-6 else 1e-6
                lower=historic_lower if time_bound=='historic' else 1e-6
                for model,nk in enumerate(NKAPPA):
                    slot=pair*14+model;info={'slot':slot,'pair':name,'model':MODELS[model],'observed_substitutions':snp,'cleaned_nt':nt,'time_bound_policy':time_bound,'historic_time_lower_bound':historic_lower}
                    if snp==0:info.update({'status':'unidentifiable_zero_substitutions','optimizer_executed':False,'result18':None});rows.append(info);continue
                    if lower>=3.:info.update({'status':'degenerate_time_box_unavailable','optimizer_executed':False,'result18':None});rows.append(info);continue
                    x=[max(lower,initial_time)]+[a.initial_rate]*6;x[nk+1]=a.initial_omega
                    lo,hi=raw_bounds(nt,snp,model,time_bound)
                    optimizer.add(slot,nk+2,x,lo,hi,nt/3);info.update({'optimizer_executed':True,'initial':x,'lower':lo,'upper':hi});jobs.append(info)
            timings['host_poll']=timings['objective']=timings['host_submit']=0.
            while jobs:
                if time.perf_counter()>deadline:raise TimeoutError('Bounded development fit time exceeded')
                tick=time.perf_counter();batch=optimizer.poll(capacity=a.request_capacity);timings['host_poll']+=time.perf_counter()-tick
                if not batch:
                    if batch.completed!=len(jobs):raise RuntimeError('No queue requests but not all fits complete')
                    break
                tick=time.perf_counter()
                if derivatives=='analytic':values,gradient_log=provider.mixed_buffered(batch.slots,batch.parameters,batch.kinds,count=len(batch))
                elif a.backend=='gpu':
                    indices=t.frombuffer(batch.slots,dtype=t.int64,count=batch.count)
                    x=t.frombuffer(batch.parameters,dtype=t.float64,count=batch.count*7).reshape(-1,7)
                    result=provider.objectives(indices,x);values=(C.c_double*len(batch)).from_address(result.data_ptr())
                elif hasattr(provider,'objectives_buffered'):values=provider.objectives_buffered(batch.slots,batch.parameters,batch.count)
                else:
                    request_rows=batch.rows();values=provider.objectives([r[0] for r in request_rows],[r[1] for r in request_rows])
                timings['objective']+=time.perf_counter()-tick
                tick=time.perf_counter()
                if derivatives=='analytic':optimizer.submit_mixed(batch,values,gradient_log)
                else:optimizer.submit(batch,values)
                timings['host_submit']+=time.perf_counter()-tick;wave_count+=1;requests+=len(batch)
            result_rows=optimizer.results();returned_gradient_summary=None
            if derivatives=='analytic':
                from publication_returned_gradient import audit_returned_gradients
                tick=time.perf_counter()
                returned_gradient_summary=audit_returned_gradients(provider,jobs,result_rows,a.request_capacity)
                timings['returned_point_gradient_audit']=time.perf_counter()-tick
            for info,result in zip(jobs,result_rows):
                finite=all(math.isfinite(v) for v in result);inside=all(info['lower'][i]<=result[i]<=info['upper'][i] for i in range(NKAPPA[info['slot']%14]+2))
                passed=finite and inside and result[14]==1 and 0.<=result[10]<=1e-6 and 0.<=result[17]<=1e-6 and result[7]<=result[8]
                info.update({'result18':[v if math.isfinite(v) else None for v in result],'status':'finite_coordinate_gate_passed' if passed else 'quality_failed','quality_passed':passed,
                    'iteration_limit_reached':result[11]>=a.max_iterations,
                    'returned_point_stationarity_certified':False});rows.append(info)
                info['active_parameter_bound_hits']=[{'parameter':i,'side':side} for i in range(NKAPPA[info['slot']%14]+2) for side,bound in [('lower',info['lower'][i]),('upper',info['upper'][i])] if abs(result[i]-bound)<=4*math.ulp(bound)] if finite else []
            rows.sort(key=lambda r:r['slot']);(a.output/'models.json').write_text(json.dumps(rows,ensure_ascii=False,indent=None,allow_nan=False)+'\n',encoding='utf-8')
            output_contract=None
            if a.complete_ma:
                from publication_output import write_complete_ma
                tick=time.perf_counter();output_contract=write_complete_ma(prepared,rows,a.output,a.backend,provider)
                timings['MA_statistics_rescoring_output']=time.perf_counter()-tick
            manifest={'schema':'kakscuda.publication_development_fit/1','complete':len(rows)==len(records)*14,'backend':a.backend,
                'count':len(records),'requested_pairs':prepared.ids,'input':str(a.input.resolve()),'input_sha256':sha(a.input),
                'model_contract':'pure-rates-strict-positive-v1','optimizer_contract':binaries['optimizer']['build']['policy'],'derivatives':derivatives,'initial_time':initial_time,'initial_rate':a.initial_rate,'initial_omega':a.initial_omega,'time_bound_policy':time_bound,
                'frequency_policy':frequency_record,
                'max_iterations':a.max_iterations,'max_seconds':a.max_seconds,'request_capacity':a.request_capacity,'batch_tasks':a.batch_tasks,'gpu_cache_buckets':getattr(a,'gpu_cache_buckets',64) if a.backend=='gpu' and derivatives=='analytic' else None,
                'CPU_topology':topo,'CPU_workers':a.workers if a.backend=='cpu' else 1,'sources':closure,'binaries':binaries,
                'python':sys.version,'platform':platform.platform(),'timings_seconds':timings,'waves':wave_count,'objective_requests':requests,
                'fit_quality_passed':all(r.get('quality_passed',True) for r in rows),'unavailable_slots':sum(not r['optimizer_executed'] for r in rows),
                'scope':'Development model fitting, not final MA tool; no legacy equality/global optimality/identifiability certificate'}
            manifest['iteration_capped_model_count']=sum(r.get('iteration_limit_reached',False) for r in rows)
            manifest['returned_point_gradient_audit']=returned_gradient_summary
            manifest['validation_authorization']=validation_ticket
            manifest['science_configuration']=science_configuration
            manifest['execution_configuration']=execution_configuration
            manifest['complete_MA_output']=output_contract
            if a.complete_ma:manifest['scope']='Experimental complete coherent MA tool; fourteen candidates and explicit unavailable rows, same optimizer/objective CPU/GPU. No legacy equality/global optimality/uncertainty claim.'
            manifest['GPU_stream_count']=0 if a.backend=='cpu' else a.stream_count
            manifest['GPU_schedule']=None if a.backend=='cpu' else a.schedule
            manifest['GPU_cold_reuse']=False if a.backend=='cpu' else True
            manifest['GPU_event_instrumentation']=False
            manifest['GPU_reduced_candidate']=None if a.backend=='cpu' else ('observed_output_shared_plus_structural587' if a.observed_layout=='shared' else 'observed_output_packed_plus_structural587')
            manifest['GPU_selected_vjp_candidate']=None if a.backend=='cpu' else 'structural587'
            manifest['backend_provenance']=provider.provenance() if a.backend=='cpu' else provider.graph_manifest()
            if a.backend=='gpu':
                manifest['GPU_observed_layout']=a.observed_layout
                manifest['GPU_observed_output_installed']=True
                manifest['peak_allocated_bytes']=provider.peak_allocated_bytes
                manifest['CUDA_runtime_engine']='CUDA Driver API; single context exact observed-output plus packed observed-adjoint and structural587 VJP; cold-result reuse; no Torch import; no CUDA timing events'
                manifest['GPU_stream_count']=a.stream_count
                manifest['selected_GPU']=selected;manifest['GPU_snapshot']=gpu_inventory
            # The saved prototype scope includes input/preparation/fit/model
            # JSON write and runtime provenance hashing. Source snapshots and
            # telemetry startup/stop are outside it. Final MA timing is separate.
            manifest['timing_scope']='Input read through prepared data, backend setup, fitting, quality/model JSON write, runtime provenance hashing; excludes pre-run snapshots, telemetry startup/stop, and final manifest write. No MA output yet.'
            if a.complete_ma:manifest['timing_scope']='Input read through preparation, backend setup, fourteen-model fitting, quality/model output, statistics, CPU final MA rescoring, TSV/JSON write and runtime provenance hashing; excludes pre-run source/binary snapshots, telemetry startup/stop, source-change audit and final manifest write.'
            manifest['wall_seconds']=time.perf_counter()-started
            manifest['source_change_audit']=verify_source_snapshot(closure)
            if not manifest['source_change_audit']['PASS']:raise RuntimeError('A captured Python source changed during the run; result is not a frozen-source benchmark')
            manifest['binary_change_audit']={'PASS':all(sha(item['path'])==item['sha256'] for item in binaries.values()),'scope':'Compare pinned on-disk binaries with pre-run hashes'}
            if not manifest['binary_change_audit']['PASS']:raise RuntimeError('A pinned binary changed during the run')
            from product_v1_engine_runtime import post_run_check
            manifest['frozen_configuration_post_run_audit']=post_run_check(validation_ticket)
            manifest.update(schema='kaksforge.product_robust_chunk/1',mode='robust',method='MA',genetic_code=1,product_engine_asset_identity=validation_ticket['engine_identity'],historical_freeze_used_as_authorization=False)
            dump(a.output/'run_manifest.json',manifest)
    except BaseException as e:
        failure={'error':str(e),'complete':False,'models_returned':rows,'seconds':time.perf_counter()-started,'timings_seconds':timings,'waves':wave_count,'requests':requests}
        if a.backend=='gpu' and 'provider' in locals():
            try:failure['GPU_execution_diagnostics']=provider.graph_manifest()
            except Exception as diagnostic_error:failure['diagnostic_error']=str(diagnostic_error)
        dump(a.output/'failure.json',failure);raise
    finally:
        if monitor:dump(a.output/'telemetry_summary.json',monitor.summary())
    print(json.dumps({'complete':manifest['complete'],'quality_passed':manifest['fit_quality_passed'],'count':manifest['count'],'wall_seconds':manifest['wall_seconds'],'waves':wave_count,'requests':requests}))
    return 0 if manifest['fit_quality_passed'] else 3
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'Codex_test/benchmark_1000/benchmark.axt');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('cpu','gpu'),required=True);p.add_argument('--count',type=int);p.add_argument('--workers',type=int,default=16);p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-tasks',type=int,default=512);p.add_argument('--request-capacity',type=int,default=16384);p.add_argument('--max-iterations',type=int,default=500);p.add_argument('--max-seconds',type=float,default=60.)
    p.add_argument('--initial-rate',type=float,default=1.2);p.add_argument('--initial-omega',type=float,default=.4);p.add_argument('--allow-heldout',action='store_true');p.add_argument('--complete-ma',action='store_true')
    p.add_argument('--initial-time',type=float,default=.05);p.add_argument('--time-bound',choices=('historic','fixed'),default='fixed')
    p.add_argument('--gpu-cache-buckets',type=int,default=256);p.add_argument('--stream-count',type=int,choices=(1,2,4),default=1);p.add_argument('--schedule',choices=('round_robin','stable_keys'),default='round_robin')
    p.add_argument('--observed-layout',choices=('shared','original'),default='original')
    p.add_argument('--frequency-truth',type=Path);p.add_argument('--frequency-scenarios',type=Path)
    p.add_argument('--study-phase',choices=('development','validation'),default='development')
    p.add_argument('--freeze-dir',type=Path);p.add_argument('--freeze-sha256');p.add_argument('--freeze-profile')
    p.add_argument('--optimizer-dll',type=Path);p.add_argument('--derivatives',choices=('fd','analytic'),default='analytic');a=p.parse_args()
    if (a.count is not None and a.count<1) or not 1<=a.workers<=128 or not a.max_seconds>0.:raise ValueError('Invalid bounded product configuration')
    return run(a)
if __name__=='__main__':raise SystemExit(main())
