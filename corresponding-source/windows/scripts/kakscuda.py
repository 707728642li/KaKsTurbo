"""Native Windows code-1 MA tool with explicit reproduce and robust modes.

reproduce: byte-validated original CLI with exact objective cache, 8 physical
cores/16 workers. robust: experimental FP64 GPU matrix exponential with V8
zero-substitution policy and bounded refinement. Robust is not legacy QL fitted-output reproduction.
References, when supplied, are read only after the computed output is saved.
"""
from __future__ import annotations
from product_v1_platform import dll_directory
import argparse
import concurrent.futures
from contextlib import contextmanager, ExitStack
import ctypes as C
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from benchmark_cpu import ROOT, SCHEMA, affinity, dump, file_sha, kernel32, schedule_shards, table, topology, tsv_bytes, validate

MODEL_ORDER=('JC','F81','K2P','HKY','TNEF','TN','K3P','K3PUF','TIMEF','TIM','TVMEF','TVM','SYM','GTR')
QUALITY_THRESHOLD=1e-6


def optimizer_binding(dll_path):
    """Select a compatible binding from DLL exports without creating a session."""
    with ExitStack() as directories:
        for directory in (dll_path.parent,ROOT/'envs/cpu-reference/Library/bin'):
            directories.enter_context(dll_directory(str(directory)))
        library=C.CDLL(str(dll_path))
        has_window=hasattr(library,'kaks_fiber_window_metrics')
        has_zero=hasattr(library,'kaks_fiber_zero_fit_metrics')
    if has_zero and not has_window:raise RuntimeError('Zero-fit DLL lacks its required window metrics ABI')
    spec={'module':'fiber_optimizer_stagnation_v5','class':'PolishedStagnationFiberOptimizer',
          'extra_sources':[],'window_metrics':has_window,'zero_fit_metrics':has_zero}
    if has_window:
        spec.update({'module':'fiber_optimizer_stagnation_v7','class':'NearStagnationFiberOptimizer'})
        spec['extra_sources']=['fiber_optimizer_stagnation_v6.py','fiber_optimizer_stagnation_v7.py']
    if has_zero:
        spec.update({'module':'fiber_optimizer_stagnation_v8','class':'ZeroAwareFiberOptimizer'})
        spec['extra_sources'].append('fiber_optimizer_stagnation_v8.py')
    return spec


def quality_summary(metrics,pair_ids,slot_offset=0):
    """Require the unmodified source e and a measured finite-coordinate pass.

    This gate describes the finite probes; it is not a global-optimum or
    legacy fitted-reproduction certificate. All per-model metrics are saved.
    """
    if len(metrics)!=len(pair_ids)*14:raise ValueError('Quality metric count differs from requested models')
    failed=[]
    for slot,metric in enumerate(metrics):
        threshold=metric['objective_improvement_threshold'];gain=metric['remaining_coordinate_gain']
        reasons=[]
        if metric['measured']!=1:reasons.append('not_measured')
        if not math.isfinite(threshold) or threshold!=QUALITY_THRESHOLD:reasons.append('source_threshold_differs')
        if not math.isfinite(gain) or gain<0 or gain>QUALITY_THRESHOLD:reasons.append('remaining_gain_exceeds_threshold_or_invalid')
        if metric['threshold_reached']!=1:reasons.append('threshold_not_reached')
        if metric['round_budget_exceeded']!=0:reasons.append('polish_budget_exceeded')
        if metric['winner_verification_rejects']!=0:reasons.append('winner_verification_rejected')
        if reasons:
            failed.append({'slot':slot_offset+slot,'pair_id':pair_ids[slot//14],'model':MODEL_ORDER[slot%14],
                           'remaining_coordinate_gain':gain,'objective_improvement_threshold':threshold,'reasons':reasons})
    return {'schema':'kakscuda.finite_coordinate_quality/1','quality_passed':not failed,
        'models':len(metrics),'unmet_models':len(failed),'failed_models':failed,
        'source_objective_improvement_threshold':QUALITY_THRESHOLD,
        'global_optimum_claimed':False,'legacy_fitted_reproduction_claimed':False,
        'scope':'Every model receives finite box-face/fine/coarse coordinate probes after bounded polishing; raw outputs retained.'}


@contextmanager
def process_affinity():
    """Keep parsing, fitting, validation and output on the selected 8/16 set."""
    k=kernel32();handle=k.GetCurrentProcess();topo=topology(k);old=affinity(k,handle)[0]
    try:
        if not k.SetProcessAffinityMask(handle,topo['selected_mask']):raise C.WinError(C.get_last_error())
        if affinity(k,handle)[0]!=topo['selected_mask']:raise RuntimeError('CPU parent affinity verification failed')
        yield topo
    finally:
        if not k.SetProcessAffinityMask(handle,old):raise C.WinError(C.get_last_error())


def project_path(value):
    path=Path(value).resolve()
    try:path.relative_to(ROOT)
    except ValueError:raise ValueError(f"Path must be inside this project: {path}")
    return path


def parse_input(data):
    """Mirror original AXT concatenation and byte-half sequence split.

    Preserve base case and ambiguity/gaps for the original paired codon filter
    and raw GC calculation. Normalize separators/wrapping only in CPU shards.
    """
    try:text=data.decode('ascii')
    except UnicodeDecodeError:raise ValueError('AXT must contain ASCII headers and sequence bytes')
    blocks=re.split(r'\n[ \t]*\n',text.replace('\r\n','\n').replace('\r','\n').strip('\n'))
    records=[];seen=set()
    for block in blocks:
        if not block.strip():continue
        lines=block.split('\n');name=lines[0]
        if not name or '\t' in name or '\x00' in name or name in seen:
            raise ValueError(f'Empty, duplicate, or TSV-incompatible AXT identifier: {name!r}')
        body=''.join(lines[1:])
        if not body or len(body)%6 or any(c.isspace() or c=='\x00' for c in body):
            raise ValueError(f'AXT sequence data must contain two equally long codon-aligned sequences: {name}')
        middle=len(body)//2;left=body[:middle].encode('ascii');right=body[middle:].encode('ascii')
        record=name.encode('ascii')+b'\n'+left+b'\n'+right+b'\n\n'
        records.append({'pair_id':name,'left':left,'right':right,'record':record,
                        'aligned_nt_per_sequence':middle});seen.add(name)
    if not records:raise ValueError('No AXT pairs found')
    return records


def check_rows(rows,requested):
    seen={}
    for row in rows:
        name=row['Sequence'];seen[name]=seen.get(name,0)+1
        if row['Method']!='MA':raise RuntimeError(f'Unexpected estimator for {name}')
        for key,value in row.items():
            if key not in ('Sequence','Method','Model') and re.search(r'(?i)\b(?:nan|inf(?:inity)?)\b',value):
                raise ArithmeticError(f'Nonfinite output at {name}, {key}: {value}')
    if set(seen)!=set(requested) or any(value!=1 for value in seen.values()):
        missing=[name for name in requested if name not in seen]
        duplicate=[name for name,value in seen.items() if value!=1]
        raise RuntimeError(f'Output identifiers differ: missing={missing[:10]}, duplicates={duplicate[:10]}')


def run_reproduce(records,args,run_dir,manifest):
    exe=ROOT/'build/cpu-reference-cached/KaKs_Calculator.exe'
    if not exe.is_file():raise FileNotFoundError(f'Build the cached Windows CLI first: {exe}')
    requested=[r['pair_id'] for r in records];by_name={r['pair_id']:r for r in records}
    shards=schedule_shards(requested,by_name,'dynamic',64)
    jobs=[]
    for i,names in enumerate(shards):
        if not names:continue
        inp=run_dir/f'chunk_{i:04d}.axt';out=run_dir/f'chunk_{i:04d}.tsv';log=run_dir/f'chunk_{i:04d}.log'
        inp.write_bytes(b''.join(by_name[name]['record'] for name in names));jobs.append((i,names,inp,out,log))
    k=kernel32();mask=manifest['CPU_topology']['selected_mask']
    manifest.update({'CPU_executable':str(exe),'CPU_executable_sha256':file_sha(exe),
                     'CPU_workers':16,'dynamic_chunks':len(jobs),
                     'CPU_scientific_algorithm':'Original code-1 MA/GY94/QL/bounded BFGS with exact 64-entry lfun cache'})
    build=exe.parent/'build.json'
    if build.exists():manifest['CPU_build_provenance']=json.loads(build.read_text(encoding='utf-8-sig'))
    def worker(job):
        index,names,inp,out,log=job;started=time.perf_counter()
        with log.open('wb') as stream:
            proc=subprocess.Popen([str(exe),'-i',str(inp),'-o',str(out),'-m','MA','-c','1'],
                                   stdout=stream,stderr=subprocess.STDOUT)
            inherited,_=affinity(k,int(proc._handle))
            if inherited!=mask:
                proc.wait();raise RuntimeError(f'Worker {index} inherited incorrect affinity {hex(inherited)}')
            code=proc.wait()
        if code:raise RuntimeError(f'Original CLI chunk {index} failed with code {code}; see {log}')
        return {'chunk':index,'pairs':len(names),'pid':proc.pid,'affinity':hex(inherited),
                'seconds':time.perf_counter()-started,'log':str(log),'output':str(out)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        manifest['CPU_jobs']=list(pool.map(worker,jobs))
    rows=[]
    for _,_,_,out,_ in jobs:rows.extend(table(out,SCHEMA))
    check_rows(rows,requested);order={name:i for i,name in enumerate(requested)}
    return sorted(rows,key=lambda row:order[row['Sequence']])


class PreparedAxtPairs:
    """Original native filter and model preparation for one resident chunk."""
    def __init__(self,records,raw_input):
        from native_adapter import NativeAdapter
        from torch_matrix_prototype import codon_edges
        self.adapter=NativeAdapter(ROOT/'build/native-adapter-checked/kaks_adapter.dll')
        self.ids=[r['pair_id'] for r in records];self.raw_input=raw_input
        self.input_sha256=hashlib.sha256(raw_input).hexdigest()
        self.pairs=[];self.meta=[];self.values=[];self.z0=[];self.z1=[];self.weights=[];self.gc=[]
        offset=0
        try:
            for record in records:
                pair=self.adapter.prepare(record['pair_id'],record['left'],record['right']);self.pairs.append(pair)
                meta,values,z0,z1,weight,gc=pair.export(offset);offset+=len(weight)
                self.meta.append(list(meta));self.values.append(list(values));self.z0.append(list(z0))
                self.z1.append(list(z1));self.weights.append(list(weight));self.gc.append(list(gc))
        except BaseException:self.close();raise
        self.max_patterns=max(map(len,self.weights));self.edges=codon_edges()
    def close(self):
        for pair in self.pairs:pair.close()
        self.pairs=[]
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def select_device(args,run_dir):
    fields='index,uuid,name,memory.total,memory.used,utilization.gpu'
    result=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    (run_dir/'nvidia-smi-before.csv').write_text(result.stdout,encoding='utf-8')
    full=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True)
    (run_dir/'nvidia-smi-before.txt').write_text(full.stdout,encoding='utf-8')
    cards=[]
    for row in csv.reader(result.stdout.splitlines()):
        parts=[p.strip() for p in row]
        cards.append({'index':int(parts[0]),'uuid':parts[1],'name':parts[2],'total_MiB':int(parts[3]),
                      'used_MiB':int(parts[4]),'utilization_percent':int(parts[5])})
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        selectors=[part.strip() for part in visible.split(',')]
        visible_cards=[]
        for ordinal,selector in enumerate(selectors):
            matches=[card for card in cards if selector==str(card['index']) or
                     (selector.startswith('GPU-') and card['uuid'].startswith(selector))]
            if len(matches)!=1:raise ValueError('Cannot resolve CUDA_VISIBLE_DEVICES to unique physical GPUs')
            if any(card['index']==matches[0]['index'] for card in visible_cards):
                raise ValueError('Duplicate GPU in CUDA_VISIBLE_DEVICES')
            visible_cards.append(dict(matches[0],CUDA_ordinal=ordinal))
        candidates=visible_cards
    else:
        candidates=[dict(card,CUDA_ordinal=card['index']) for card in cards]
    # Reserve 2 GiB plus a conservative context/graph/input estimate.
    eligible=[c for c in candidates if c['total_MiB']-c['used_MiB']>=4096]
    if args.device!='auto':
        index=int(args.device.split(':')[1]);chosen=next((c for c in eligible if c['index']==index),None)
        if chosen is None:raise MemoryError('Selected GPU lacks the required 2 GiB reserve and estimated workspace')
    else:
        idle=[c for c in eligible if c['utilization_percent']==0 and c['used_MiB']<=100]
        if len(idle)>1:chosen=next((c for c in idle if c['index']==1),idle[0])
        elif idle:chosen=idle[0]
        else:
            spare=[c for c in eligible if c['utilization_percent']<80]
            if not spare:raise RuntimeError('No GPU currently has enough memory and compute headroom')
            chosen=min(spare,key=lambda c:(c['utilization_percent'],c['used_MiB']))
    return 'cuda:'+str(chosen['CUDA_ordinal']),chosen,cards


def run_robust(records,raw_input,args,run_dir,manifest):
    if not args.optimizer_dll.is_file():raise FileNotFoundError(f'Experimental optimizer DLL is missing: {args.optimizer_dll}')
    device,selected,cards=select_device(args,run_dir)
    # The command is a standalone process. Select by physical UUID before any
    # Torch/CUDA import, so CUDA ordinal ordering cannot touch a different card.
    os.environ['CUDA_VISIBLE_DEVICES']=selected['uuid'];device='cuda:0'
    manifest.update({'CUDA_device':device,'selected_GPU':selected,'GPU_snapshot':cards,
        'CUDA_VISIBLE_DEVICES':selected['uuid'],
        'experimental_optimizer_policy':3,'experimental_mode':True,
        'legacy_QL_fitted_reproduction_claimed':False,'GPU_backend':args.backend,
        'batch_tasks':args.batch_tasks,'GPU_telemetry_interval_ms':args.telemetry_ms})
    import torch
    torch.set_num_threads(16)
    from gpu_telemetry import GPUTelemetry
    from ma_reporting import robust_row
    binding=args.optimizer_binding
    Optimizer=getattr(importlib.import_module(binding['module']),binding['class'])
    if args.backend in ('graph-buffered','graph-pipelined'):
        from fiber_optimizer_buffered import BufferedFiberOptimizer
        if args.backend=='graph-pipelined':
            from torch_graph_likelihood_pipelined import TorchGraphBatchLikelihoodPipelined as Provider
        else:
            from torch_graph_likelihood_buffered import TorchGraphBatchLikelihoodBuffered as Provider
        class BufferedRobustOptimizer(BufferedFiberOptimizer,Optimizer):
            """Borrowed tensor requests plus the selected diagnostic binding."""
            pass
        Optimizer=BufferedRobustOptimizer
    elif args.backend=='graph-grouped':
        from torch_graph_likelihood_grouped import TorchGraphBatchLikelihoodGrouped as Provider
    else:
        from torch_graph_likelihood_fast import TorchGraphBatchLikelihoodFast as Provider
    optimizer_build=args.optimizer_dll.parent/'build.json'
    if optimizer_build.is_file():manifest['optimizer_build_provenance']=json.loads(optimizer_build.read_text(encoding='utf-8-sig'))
    monitor=None;rows=[];chunks=[];unadjusted_rows=[];reporting=[];quality_reports=[]
    manifest['zero_substitution_reporting_policy']='kakscuda.zero_substitution_NA/1'
    try:
        monitor=GPUTelemetry(run_dir,selected['uuid'],args.telemetry_ms)
        for chunk_start in range(0,len(records),1000):
            chunk_started=time.perf_counter()
            chunk_records=records[chunk_start:chunk_start+1000]
            chunk_dir=run_dir/f'gpu_chunk_{chunk_start//1000:04d}';chunk_dir.mkdir()
            timings={'native_preparation_seconds':0.,'GPU_provider_setup_seconds':0.,
                     'optimizer_setup_seconds':0.,'poll_seconds':0.,'GPU_objective_seconds':0.,
                     'submit_seconds':0.,'result_export_seconds':0.,'MA_postprocess_seconds':0.}
            tick=time.perf_counter()
            with PreparedAxtPairs(chunk_records,raw_input) as prepared:
                timings['native_preparation_seconds']=time.perf_counter()-tick
                provider=None
                try:
                    tick=time.perf_counter()
                    with Optimizer(args.optimizer_dll,max_pairs=len(chunk_records)) as optimizer:
                        for record in chunk_records:
                            optimizer.add_pair(record['pair_id'],record['left'],record['right'])
                        timings['optimizer_setup_seconds']=time.perf_counter()-tick
                        tick=time.perf_counter();provider=Provider(prepared,device,args.batch_tasks)
                        timings['GPU_provider_setup_seconds']=time.perf_counter()-tick
                        actual_uuid=str(torch.cuda.get_device_properties(provider.device).uuid)
                        if not actual_uuid.startswith('GPU-'):actual_uuid='GPU-'+actual_uuid
                        if actual_uuid!=selected['uuid']:
                            raise RuntimeError('CUDA ordinal mapping differs from selected nvidia-smi GPU; stopped before fitting')
                        torch.cuda.reset_peak_memory_stats(provider.device)
                        waves=[]
                        while True:
                            wave_started=time.perf_counter();tick=wave_started
                            batch=optimizer.poll(capacity=len(chunk_records)*14)
                            timings['poll_seconds']+=time.perf_counter()-tick
                            if not len(batch):
                                if batch.completed!=len(chunk_records)*14:raise RuntimeError('No pending requests with unfinished models')
                                break
                            tick=time.perf_counter();values=provider.objectives(batch.slots,batch.parameters)
                            timings['GPU_objective_seconds']+=time.perf_counter()-tick
                            tick=time.perf_counter();optimizer.submit(batch,values)
                            timings['submit_seconds']+=time.perf_counter()-tick
                            waves.append({'wave':len(waves)+1,'pending':len(batch),'completed':batch.completed,
                                          'poll_compute_submit_seconds':time.perf_counter()-wave_started})
                            if len(waves)==1 or len(waves)%500==0:
                                print(f"robust chunk {chunk_start//1000+1}: {batch.completed}/{len(chunk_records)*14} models, wave {len(waves)}",flush=True)
                        tick=time.perf_counter()
                        raw,status=optimizer.results();metrics=optimizer.metrics();diagnostics=optimizer.refinement_diagnostics()
                        protected_metrics=optimizer.protected_cache_metrics()
                        polish_metrics=optimizer.polish_metrics()
                        window_metrics=optimizer.window_metrics() if binding['window_metrics'] else None
                        zero_fit_metrics=optimizer.zero_fit_metrics() if binding['zero_fit_metrics'] else None
                        timings['result_export_seconds']=time.perf_counter()-tick
                        optimizer_provenance={'path':str(optimizer.path),'sha256':file_sha(optimizer.path)}
                    tick=time.perf_counter();postprocess=[];chunk_unadjusted=[];chunk_reporting=[]
                    for index,pair in enumerate(prepared.pairs):
                        row,cpu_seconds,diagnostic=pair.finalize(raw[index*14*96:(index+1)*14*96],status[index*14*4:(index+1)*14*4])
                        chunk_unadjusted.append(row);unadjusted_rows.append(row)
                        row,info=robust_row(row,prepared.values[index][1])
                        chunk_reporting.append(info);reporting.append(info)
                        values=row.rstrip('\n').split('\t')
                        if len(values)!=22:raise RuntimeError('Native MA output did not contain 22 columns')
                        rows.append(dict(zip(SCHEMA,values)))
                        postprocess.append({'pair_id':prepared.ids[index],'cpu_seconds':cpu_seconds,**diagnostic})
                    timings['MA_postprocess_seconds']=time.perf_counter()-tick
                    (chunk_dir/'model_results.f64').write_bytes(bytes(raw));(chunk_dir/'model_statuses.i32').write_bytes(bytes(status))
                    dump(chunk_dir/'policy3_diagnostics.json',diagnostics);dump(chunk_dir/'model_metrics.json',metrics)
                    dump(chunk_dir/'protected_cache_metrics.json',protected_metrics)
                    dump(chunk_dir/'polish_metrics.json',polish_metrics)
                    if window_metrics is not None:dump(chunk_dir/'window_metrics.json',window_metrics)
                    if zero_fit_metrics is not None:dump(chunk_dir/'zero_fit_metrics.json',zero_fit_metrics)
                    quality=quality_summary(polish_metrics,prepared.ids,chunk_start*14)
                    dump(chunk_dir/'quality_summary.json',quality);quality_reports.append(quality)
                    (chunk_dir/'unadjusted.MA.tsv').write_bytes(('\t'.join(SCHEMA)+'\n'+''.join(chunk_unadjusted)).encode('ascii'))
                    dump(chunk_dir/'reporting_diagnostics.json',chunk_reporting)
                    chunk_report={'pairs':len(chunk_records),'requested_pairs':prepared.ids,'waves':waves,
                        'graph':provider.graph_manifest(),'optimizer_DLL':optimizer_provenance,
                        'optimizer_implementation_revision':manifest.get('optimizer_build_provenance',{}).get('implementation_revision','unknown'),
                        'window_metrics_saved':window_metrics is not None,
                        'zero_fit_metrics_saved':zero_fit_metrics is not None,
                        'MA_postprocess':postprocess,'quality_summary':quality,
                        'timings':timings,'native_adapter_sha256':file_sha(prepared.adapter.path),
                        'torch_version':torch.__version__,'CUDA_version':torch.version.cuda,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(provider.device),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(provider.device)}
                    chunk_report['chunk_seconds']=time.perf_counter()-chunk_started
                    dump(chunk_dir/'run_manifest.json',chunk_report);chunks.append(chunk_report)
                finally:
                    if provider is not None:
                        provider.native_host.close();del provider
                        torch.cuda.empty_cache()
        manifest['GPU_chunks']=chunks
        (run_dir/'unadjusted.MA.tsv').write_bytes(('\t'.join(SCHEMA)+'\n'+''.join(unadjusted_rows)).encode('ascii'))
        dump(run_dir/'reporting_diagnostics.json',reporting)
        manifest['zero_substitution_reported_pairs']=sum(item['uninformative_zero_substitutions'] for item in reporting)
        quality={'schema':'kakscuda.finite_coordinate_quality/1','models':len(records)*14,
                 'quality_passed':all(item['quality_passed'] for item in quality_reports),
                 'unmet_models':sum(item['unmet_models'] for item in quality_reports),
                 'failed_models':[model for item in quality_reports for model in item['failed_models']],
                 'source_objective_improvement_threshold':QUALITY_THRESHOLD,
                 'global_optimum_claimed':False,'legacy_fitted_reproduction_claimed':False}
        dump(run_dir/'quality_summary.json',quality)
        manifest['quality_passed']=quality['quality_passed'];manifest['unmet_quality_models']=quality['unmet_models']
    finally:
        if monitor:
            manifest['GPU_telemetry']=monitor.summary()
            manifest['GPU_telemetry']['sampling_note']='NVML device sampling window, not kernel occupancy. Tool wall time includes telemetry startup/stop.'
    check_rows(rows,[r['pair_id'] for r in records])
    return rows


def validate_reference(reference_path,output,requested):
    """Reference errors are diagnostics after the calculated output is saved."""
    reference_rows=table(reference_path,SCHEMA);counts={}
    for row in reference_rows:counts[row['Sequence']]=counts.get(row['Sequence'],0)+1
    missing=[name for name in requested if name not in counts]
    duplicates=[name for name,count in counts.items() if count>1]
    if missing or duplicates:
        return {'passed':False,'kind':'invalid_reference','missing_requested_pairs':missing,
                'duplicate_identifiers':duplicates,'computed_output_retained':True}
    references={row['Sequence']:row for row in reference_rows}
    report,_=validate(references,[output],requested)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Project-local codon-aligned AXT')
    parser.add_argument('--mode',choices=['reproduce','robust'],required=True)
    parser.add_argument('--output',type=Path,required=True,help='New MA TSV; logs go to <output>.run/')
    parser.add_argument('--reference',type=Path,help='Optional TSV comparison, read only after computation')
    parser.add_argument('--device',choices=['auto','cuda:0','cuda:1'],default='auto',help='Physical nvidia-smi GPU index; auto follows project sharing rules')
    parser.add_argument('--backend',choices=['graph-fast','graph-grouped','graph-buffered','graph-pipelined'],default='graph-buffered')
    parser.add_argument('--optimizer-dll',type=Path,default=ROOT/'build/fiber-optimizer-stagnation-v8-final/fiber_optimizer.dll',
                        help='Experimental compatible DLL; default V8-final with zero-SNP policy, window metrics and bounded coordinate quality')
    parser.add_argument('--batch-tasks',type=int,default=256)
    parser.add_argument('--telemetry-ms',type=int,default=500)
    parser.add_argument('--validate-only',action='store_true',help='Validate AXT without CPU/GPU fitting or output writes')
    args=parser.parse_args(argv)
    if args.batch_tasks<1 or args.telemetry_ms<0 or 0<args.telemetry_ms<100:parser.error('Batch size must be positive; telemetry must be >=100 ms or 0')
    args.input=project_path(args.input);args.output=project_path(args.output)
    if args.reference:args.reference=project_path(args.reference)
    args.optimizer_dll=project_path(args.optimizer_dll)
    if os.name!='nt':raise RuntimeError('Native Windows is required')
    if args.validate_only:return execute(args,None)
    with process_affinity() as topo:return execute(args,topo)


def execute(args,topo):
    started=time.perf_counter();data=args.input.read_bytes();records=parse_input(data)
    if args.validate_only:
        print(json.dumps({'input':str(args.input),'pairs':len(records),'input_sha256':hashlib.sha256(data).hexdigest(),
                          'computation_executed':False}));return 0
    run_dir=project_path(str(args.output)+'.run')
    if args.output.exists() or run_dir.exists():raise ValueError('Output or run directory already exists; choose a new output')
    if args.mode=='robust':args.optimizer_binding=optimizer_binding(args.optimizer_dll)
    args.output.parent.mkdir(parents=True,exist_ok=True);run_dir.mkdir()
    sources=['kakscuda.py','benchmark_cpu.py']
    if args.mode=='robust':sources+=['fiber_optimizer.py','fiber_optimizer_stable.py','fiber_optimizer_stagnation.py',
        'fiber_optimizer_stagnation_v4.py','fiber_optimizer_stagnation_v5.py','native_adapter.py','torch_matrix_prototype.py',
        'torch_graph_likelihood.py','torch_graph_likelihood_fast.py','gpu_telemetry.py','ma_reporting.py']
    if args.mode=='robust':sources+=args.optimizer_binding['extra_sources']
    if args.mode=='robust' and args.backend=='graph-grouped':sources.append('torch_graph_likelihood_grouped.py')
    if args.mode=='robust' and args.backend in ('graph-buffered','graph-pipelined'):
        sources+=['fiber_optimizer_buffered.py',
                  'torch_graph_likelihood_pipelined.py' if args.backend=='graph-pipelined' else 'torch_graph_likelihood_buffered.py']
    snapshot=run_dir/'source_snapshot';snapshot.mkdir();source_hashes={}
    for name in sources:
        source=ROOT/'scripts'/name;shutil.copyfile(source,snapshot/name);source_hashes[name]=file_sha(source)
    manifest={'schema':'kakscuda.tool_run/1','mode':args.mode,'complete':False,'input':str(args.input),
        'input_sha256':hashlib.sha256(data).hexdigest(),'count':len(records),'genetic_code':1,'method':'MA',
        'model_count':14,'model_order':MODEL_ORDER,'output':str(args.output),'native_windows':True,'python':sys.version,
        'source_snapshot_sha256':source_hashes,'reference_used_during_computation':False,
        'CPU_topology':topo,'affinity_scope':'Input parsing through computed output and optional reference diagnostics',
        'input_normalization':'Header/base bytes and request order retained; CPU shards normalize separators/wrapping only',
        'timing_scope':'Input parsing, snapshots, preparation, compute, transfers, diagnostics and MA output; optional reference validation excluded'}
    if args.mode=='robust':manifest['optimizer_binding']=args.optimizer_binding
    dump(run_dir/'run_manifest.json',manifest)
    try:
        print('reproduce: original Windows FP64/QL MA with exact cache.' if args.mode=='reproduce' else
              'robust: EXPERIMENTAL FP64 matrix exponential + policy3; does not reproduce legacy QL fitted output.',flush=True)
        rows=run_reproduce(records,args,run_dir,manifest) if args.mode=='reproduce' else run_robust(records,data,args,run_dir,manifest)
        check_rows(rows,[r['pair_id'] for r in records]);args.output.write_bytes(tsv_bytes(rows,SCHEMA))
        manifest.update({'complete':True,'end_to_end_seconds':time.perf_counter()-started,
            'output_sha256':file_sha(args.output),'output_rows':len(rows),'all_22_columns_present':True})
        dump(run_dir/'run_manifest.json',manifest)
        # No reference bytes or values are available to either estimator above.
        if args.reference:
            report=validate_reference(args.reference,args.output,[r['pair_id'] for r in records])
            dump(run_dir/'reference_validation.json',report)
            manifest['reference_validation_passed']=report['passed'];manifest['reference_sha256']=file_sha(args.reference)
            dump(run_dir/'run_manifest.json',manifest)
        print(json.dumps({'output':str(args.output),'log_directory':str(run_dir),'mode':args.mode,
                          'pairs':len(rows),'seconds':manifest['end_to_end_seconds'],
                          'quality_passed':manifest.get('quality_passed'),
                          'unmet_quality_models':manifest.get('unmet_quality_models'),
                          'reference_validation_passed':manifest.get('reference_validation_passed')}))
        if manifest.get('quality_passed') is False:return 3
        return 2 if manifest.get('reference_validation_passed') is False else 0
    except BaseException as error:
        manifest.update({'failure':{'type':type(error).__name__,'message':str(error),
                                  'stage':'after_computation_diagnostics' if manifest['complete'] else 'calculation'},
                         'elapsed_seconds':time.perf_counter()-started})
        dump(run_dir/'run_manifest.json',manifest);raise


if __name__=='__main__':raise SystemExit(main())
