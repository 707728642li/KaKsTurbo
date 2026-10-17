"""New pure-rate/strict-positive FP64 objective, distinct from frozen V8.

Resident pattern/frequency tensors, guarded T18, grouped CUDA graphs and reused
CPU buffers are inherited with their recorded sources. New semantics are
explicit: pi>0; no repeated rate transformation; simultaneous repeated slots.
No optimizer, approximate likelihood clipping, or lower precision is hidden.
"""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import time
from torch_graph_likelihood_buffered import TorchGraphBatchLikelihoodBuffered,ROOT

class PublicationGPULikelihood(TorchGraphBatchLikelihoodBuffered):
    def __init__(self,prepared,device='cuda:1',batch_tasks=512,**options):
        super().__init__(prepared,device,batch_tasks,host_dll=ROOT/'build/publication-graph-host-v1/torch_graph_host.dll',**options)
        d=self.native_host.dll
        d.kaks_publication_validate.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32,C.c_char_p,C.c_uint32]
        d.kaks_publication_validate.restype=C.c_int
        def validate(indices,x):
            self.native_host._check(d.kaks_publication_validate(self.native_host.context,indices.data_ptr(),x.data_ptr(),len(indices),self.native_host.error,len(self.native_host.error)))
        self.native_host.validate=validate

    def _prepare_host_bounds(self):
        # Same triangle bound as the frozen graph, retaining ALL positive pi.
        def coefficients(pi):
            live=[p>0. for p in pi];columns=[[0.]*12 for _ in range(61)];mr=[0.]*12
            for i,j,rate,syn in self.prepared.edges:
                k=rate+(0 if syn else 6);symmetric=math.sqrt(pi[i]*pi[j]) if live[i] and live[j] else 0.
                if live[i]:columns[i][k]+=pi[j]+symmetric
                if live[j]:columns[j][k]+=pi[i]+symmetric
                mr[k]+=(2.*pi[i])*pi[j]
            return [max(row[k] for row in columns) for k in range(12)],mr
        equal=coefficients([1./61]*61)
        self.host_bounds=[(equal,coefficients(values[2:63])) for values in self.prepared.values]

    def _matrix_inputs(self,indices,x):
        t=self.torch;pi=self.pi[indices,:61]
        omega=x.gather(1,(self.nkappa[indices]+1)[:,None])[:,0]
        raw=t.where(self.active_parameter[indices],x[:,1:7],t.ones_like(x[:,1:7]))
        rates=raw.gather(1,self.rate_map[indices]);q,mr=self._q_mr(rates,pi,omega)
        live=pi>0.;sqrt_pi=t.where(live,pi,t.ones_like(pi)).sqrt()
        s=(q*sqrt_pi[:,:,None])/sqrt_pi[:,None,:]
        lower=t.tril(s,diagonal=-1)
        symmetric=lower+lower.transpose(-1,-2)+t.diag_embed(q.diagonal(dim1=-2,dim2=-1))
        live_outer=live[:,:,None]&live[:,None,:]
        symmetric=t.where(live_outer,symmetric,t.zeros_like(symmetric))
        a=(symmetric*x[:,0,None,None])/mr[:,None,None]
        return a,pi,sqrt_pi,live_outer

    def objectives(self,slots,parameters):
        t=self.torch
        if len(slots)!=len(parameters):raise ValueError('Slot/parameter count mismatch')
        count=len(slots)
        if not count:return t.empty(0,dtype=t.float64,device='cpu')
        started=time.perf_counter()
        if not isinstance(slots,t.Tensor) or not isinstance(parameters,t.Tensor):raise ValueError('Expected CPU tensor requests')
        if slots.device.type!='cpu' or slots.dtype!=t.int64 or slots.shape!=(count,) or not slots.is_contiguous():raise ValueError('Expected contiguous CPU int64 slots')
        if parameters.device.type!='cpu' or parameters.dtype!=t.float64 or parameters.shape!=(count,7) or not parameters.is_contiguous():raise ValueError('Expected contiguous CPU FP64 x7')
        self.native_host.validate(slots,parameters)
        if count>len(self.group_order):
            self.group_order=t.empty(count,dtype=t.int64,device='cpu');self.group_levels=t.empty(count,dtype=t.int32,device='cpu')
        self.host_marshal_validate_seconds+=time.perf_counter()-started
        started=time.perf_counter()
        self.native_host._check(self.native_host.dll.gy94_graph_host_group(
            self.native_host.context,slots.data_ptr(),parameters.data_ptr(),count,
            self.max_squarings,self.adaptive_squarings,self.group_order.data_ptr(),
            self.group_levels.data_ptr(),self.group_offsets,C.byref(self.native_host.maximum),
            self.native_host.error,len(self.native_host.error)))
        self.native_group_seconds+=time.perf_counter()-started
        order=self.group_order[:count];grouped_indices=slots.index_select(0,order);grouped_x=parameters.index_select(0,order)
        values=t.empty(count,dtype=t.float64,device='cpu')
        with t.inference_mode():
            for squarings in range(self.max_squarings+1):
                begin,end=self.group_offsets[squarings],self.group_offsets[squarings+1]
                self.grouped_row_histogram[squarings]+=end-begin
                for offset in range(begin,end,self.batch_tasks):
                    stop=min(end,offset+self.batch_tasks);indices=grouped_indices[offset:stop];x=grouped_x[offset:stop];local_order=order[offset:stop]
                    local_count=stop-offset
                    size=(1<<(local_count-1).bit_length()) if local_count<=self.bucket_quantum else ((local_count+self.bucket_quantum-1)//self.bucket_quantum)*self.bucket_quantum
                    bucket=self.buckets.get((size,squarings))
                    if bucket is None:
                        initial_indices=t.empty(size,dtype=t.int64,device=self.device);initial_x=t.empty((size,7),dtype=t.float64,device=self.device)
                        initial_indices[:local_count].copy_(indices);initial_x[:local_count].copy_(x)
                        if local_count<size:
                            initial_indices[local_count:].copy_(initial_indices[:1].expand(size-local_count));initial_x[local_count:].copy_(initial_x[:1].expand(size-local_count,7))
                        bucket=self._new_bucket(size,squarings,initial_indices,initial_x);transferred=local_count*64
                    else:
                        bucket['host_indices'][:local_count].copy_(indices);bucket['host_x'][:local_count].copy_(x)
                        if local_count<size:
                            bucket['host_indices'][local_count:].copy_(indices[:1].expand(size-local_count));bucket['host_x'][local_count:].copy_(x[:1].expand(size-local_count,7))
                        bucket['indices'].copy_(bucket['host_indices'],non_blocking=True);bucket['x'].copy_(bucket['host_x'],non_blocking=True);transferred=size*64
                    if bucket['graph'] is None:result=self._forward(bucket['indices'],bucket['x'],squarings)
                    else:bucket['graph'].replay();self.graph_replays+=1;result=bucket['result']
                    host_result=result[:local_count].cpu().contiguous();bad,maximum=self.native_host.analyze(host_result)
                    self.maximum_observed_norm=max(self.maximum_observed_norm,maximum)
                    if bad:
                        if self.fallback=='raise':raise ArithmeticError('Grouped T18 scaling guard exceeded')
                        fallback=self._fallback_values([int(indices[i]) for i in bad],[x[i].tolist() for i in bad])
                        for i,value in zip(bad,fallback):
                            if not math.isfinite(value):raise ArithmeticError('Nonfinite guarded fallback likelihood')
                            host_result[i,0]=value
                    values.index_copy_(0,local_order,host_result[:,0]);self.parameter_transfer_bytes+=transferred;self.objective_transfer_bytes+=local_count*24
                    self.grouped_padded_rows+=size;self.grouped_square_products+=size*squarings;self.grouped_T18_products+=size*(5+squarings);self.grouped_graph_calls+=1
        self.requests+=count
        return values

    def graph_manifest(self):
        manifest=super().graph_manifest()
        manifest.update({'schema':'kakscuda.publication_gpu_objective/2','model_contract':'pure-rates-strict-positive-v1',
            'support_cutoff':0.,'repeated_slot_requests':True,'optimizer_changed':False,
            'scope':'objective backend only; distinct model contract from historical reproduction and frozen V8'})
        return manifest
