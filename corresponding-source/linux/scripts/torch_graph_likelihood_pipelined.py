"""Independent v5: same grouped graph arithmetic with one synchronization per wave.

Graph arithmetic is inherited from v1/v2. Only execution order changes; output
is scattered back to request order before native optimizer submission.
"""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import time

from torch_graph_likelihood_fast import TorchGraphBatchLikelihoodFast

ROOT=Path(__file__).resolve().parents[1]


class TorchGraphBatchLikelihoodPipelined(TorchGraphBatchLikelihoodFast):
    def __init__(self,prepared,device="cuda:1",batch_tasks=256,max_squarings=6,
                 use_graphs=True,fallback="torch",adaptive_squarings=True,bucket_quantum=32,
                 host_dll=None):
        import torch as t
        # Pin v2's otherwise implicit CPU host coefficient factory, without
        # changing the frozen v2 module. All selected GPU factories are explicit.
        with t.device("cpu"):
            super().__init__(prepared,device,batch_tasks,max_squarings,use_graphs,
                fallback,adaptive_squarings,bucket_quantum,
                host_dll or ROOT/"build/torch-graph-host-grouped/torch_graph_host.dll")
        self.group_order=t.empty(self.task_count,dtype=t.int64,device="cpu")
        self.group_levels=t.empty(self.task_count,dtype=t.int32,device="cpu")
        self.group_offsets=(C.c_uint32*18)()
        d=self.native_host.dll
        d.gy94_graph_host_group.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32,
            C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_double),
            C.c_char_p,C.c_uint32]
        d.gy94_graph_host_group.restype=C.c_int
        self.grouped_row_histogram=[0]*(max_squarings+1)
        self.grouped_padded_rows=0
        self.grouped_square_products=0
        self.grouped_T18_products=0
        self.grouped_graph_calls=0
        self.native_group_seconds=0.
        self.wave_indices=t.empty(self.task_count,dtype=t.int64,device="cpu",pin_memory=True)
        self.wave_x=t.empty((self.task_count,7),dtype=t.float64,device="cpu",pin_memory=True)
        self.wave_result=t.empty((self.task_count,3),dtype=t.float64,device="cpu",pin_memory=True)
        self.wave_syncs=0
        self.failed=False

    def _new_bucket(self,*args):
        # v2 creates pinned host staging with implicit device; force its CPU
        # factories locally even if a caller changes the global default device.
        with self.torch.device("cpu"):
            return super()._new_bucket(*args)

    def objectives(self,slots,parameters):
        if self.failed:
            raise RuntimeError("Pipelined provider cannot be reused after a failed wave")
        try:
            return self._objectives_wave(slots,parameters)
        except BaseException:
            self.failed=True
            # An interrupted capture/copy may have queued work on a helper
            # stream. Drain this process's device context before releasing
            # pinned buffers, and make failure terminal for this provider.
            try:self.torch.cuda.synchronize(self.device)
            except Exception:pass
            raise

    def _objectives_wave(self,slots,parameters):
        if len(slots)!=len(parameters):raise ValueError("Slot and parameter counts differ")
        count=len(slots)
        if not count:return []
        if count>self.task_count:raise ValueError("Request count exceeds available unique slots")
        t=self.torch
        started=time.perf_counter()
        if not isinstance(slots,t.Tensor) or not isinstance(parameters,t.Tensor):
            raise ValueError("Buffered provider requires CPU tensor requests")
        if slots.device.type!="cpu" or slots.dtype!=t.int64 or not slots.is_contiguous():
            raise ValueError("Buffered slot IDs must be contiguous CPU int64")
        if parameters.device.type!="cpu" or parameters.dtype!=t.float64 or not parameters.is_contiguous():
            raise ValueError("Buffered parameters must be contiguous CPU FP64")
        host_indices=slots
        host_x=parameters
        if host_indices.shape!=(count,) or host_x.shape!=(count,7):
            raise ValueError("Parameters must contain seven doubles per slot")
        self.native_host.validate(host_indices,host_x)
        self.host_marshal_validate_seconds+=time.perf_counter()-started
        started=time.perf_counter()
        self.native_host._check(self.native_host.dll.gy94_graph_host_group(
            self.native_host.context,host_indices.data_ptr(),host_x.data_ptr(),count,
            self.max_squarings,self.adaptive_squarings,self.group_order.data_ptr(),
            self.group_levels.data_ptr(),self.group_offsets,C.byref(self.native_host.maximum),
            self.native_host.error,len(self.native_host.error)))
        self.native_group_seconds+=time.perf_counter()-started
        order=self.group_order[:count]
        grouped_indices=host_indices.index_select(0,order)
        grouped_x=host_x.index_select(0,order)
        values=t.empty(count,dtype=t.float64,device="cpu")
        # Every row has its own immutable pinned staging slice until the
        # complete wave finishes. Bucket reuse cannot overwrite pending H2D.
        self.wave_indices[:count].copy_(grouped_indices)
        self.wave_x[:count].copy_(grouped_x)
        with t.inference_mode():
            for squarings in range(self.max_squarings+1):
                begin,end=self.group_offsets[squarings],self.group_offsets[squarings+1]
                self.grouped_row_histogram[squarings]+=end-begin
                for offset in range(begin,end,self.batch_tasks):
                    stop=min(end,offset+self.batch_tasks)
                    indices=grouped_indices[offset:stop];x=grouped_x[offset:stop]
                    local_order=order[offset:stop]
                    local_count=stop-offset
                    size=(1<<(local_count-1).bit_length()) if local_count<=self.bucket_quantum else (
                        (local_count+self.bucket_quantum-1)//self.bucket_quantum)*self.bucket_quantum
                    bucket=self.buckets.get((size,squarings))
                    if bucket is None:
                        initial_indices=t.empty(size,dtype=t.int64,device=self.device)
                        initial_x=t.empty((size,7),dtype=t.float64,device=self.device)
                        initial_indices[:local_count].copy_(indices);initial_x[:local_count].copy_(x)
                        if local_count<size:
                            initial_indices[local_count:].copy_(initial_indices[:1].expand(size-local_count))
                            initial_x[local_count:].copy_(initial_x[:1].expand(size-local_count,7))
                        bucket=self._new_bucket(size,squarings,initial_indices,initial_x)
                        transferred=local_count*64
                    else:
                        bucket['indices'][:local_count].copy_(self.wave_indices[offset:stop],non_blocking=True)
                        bucket['x'][:local_count].copy_(self.wave_x[offset:stop],non_blocking=True)
                        if local_count<size:
                            bucket['indices'][local_count:].copy_(bucket['indices'][:1].expand(size-local_count))
                            bucket['x'][local_count:].copy_(bucket['x'][:1].expand(size-local_count,7))
                        transferred=local_count*64
                    if bucket['graph'] is None:
                        result=self._forward(bucket['indices'],bucket['x'],squarings)
                    else:
                        bucket['graph'].replay();self.graph_replays+=1;result=bucket['result']
                    # Same-stream D2H is ordered before this bucket's next
                    # replay. Each chunk writes a distinct pinned CPU slice.
                    self.wave_result[offset:stop].copy_(result[:local_count],non_blocking=True)
                    self.parameter_transfer_bytes+=transferred
                    self.objective_transfer_bytes+=local_count*24
                    self.grouped_padded_rows+=size
                    self.grouped_square_products+=size*squarings
                    self.grouped_T18_products+=size*(5+squarings)
                    self.grouped_graph_calls+=1
            # The optimizer receives results only after all copies complete.
            t.cuda.current_stream(self.device).synchronize()
            self.wave_syncs+=1
            host_result=self.wave_result[:count]
            bad,maximum=self.native_host.analyze(host_result)
            self.maximum_observed_norm=max(self.maximum_observed_norm,maximum)
            if bad:
                if self.fallback=='raise':
                    raise ArithmeticError("Pipelined T18 scaling/finite guard exceeded at "+str(
                        [(int(grouped_indices[i]),host_result[i,2].item()) for i in bad]))
                fallback=self._fallback_values([int(grouped_indices[i]) for i in bad],
                                               [grouped_x[i].tolist() for i in bad])
                for i,value in zip(bad,fallback):
                    if not math.isfinite(value):raise ArithmeticError("Nonfinite pipelined fallback likelihood")
                    host_result[i,0]=value
            values.index_copy_(0,order,host_result[:,0])
        self.requests+=count
        return values

    def graph_manifest(self):
        report=super().graph_manifest()
        report.update({'schema':'kakscuda.torch_graph_likelihood_pipelined/5',
            'grouping':'native stable whole-wave permutation by individual host squaring bound',
            'native_group_seconds':self.native_group_seconds,
            'grouped_row_histogram':self.grouped_row_histogram,
            'grouped_padded_rows':self.grouped_padded_rows,
            'grouped_square_matrix_products':self.grouped_square_products,
            'grouped_T18_plus_square_matrix_products':self.grouped_T18_products,
            'grouped_graph_calls':self.grouped_graph_calls,
            'output_order':'scatter to original request row before submission',
            'CPU_host_factories_explicit':True,
            'wave_synchronizations':self.wave_syncs,
            'copy_schedule':'same-stream distinct pinned wave slices; synchronize before CPU analysis/scatter',
            'request_API':'borrowed contiguous CPU tensors; returns contiguous CPU objective tensor'})
        return report
