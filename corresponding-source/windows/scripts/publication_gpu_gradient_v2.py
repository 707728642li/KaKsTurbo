"""Independent bounded-cache execution policy for the verified GPU gradient.

The v1 Q/T18/AD arithmetic is inherited without edits. Power-of-two bucket
shapes and an LRU count bound prevent unbounded CUDA Graph private-pool growth.
Before capture, older pools are released until the same conservative workspace
estimate and2GiB reserve fit. If necessary the next retry uses smaller chunks.
"""
from __future__ import annotations
from collections import OrderedDict,deque
import hashlib
from pathlib import Path
import threading
from publication_gpu_gradient import PublicationGPUGradient as VerifiedGPUGradient

class GraphWorkspaceBudgetError(MemoryError):
    def __init__(self,message,suggested_rows):super().__init__(message);self.suggested_rows=suggested_rows

class LRUBuckets(OrderedDict):
    def get(self,key,default=None):
        value=super().get(key,default)
        if key in self:self.move_to_end(key)
        return value

class PublicationGPUGradient(VerifiedGPUGradient):
    def __init__(self,prepared,device='cuda:1',batch_tasks=512,max_cached_buckets=8,
                 bucket_quantum=None,**options):
        if not isinstance(max_cached_buckets,int) or max_cached_buckets<1:raise ValueError('Use at least one cached bucket')
        # Every local_count <= batch_tasks takes v1's power-of-two branch.
        super().__init__(prepared,device,batch_tasks,bucket_quantum=max(1,batch_tasks),**options)
        self.gradient_buckets=LRUBuckets();self.max_cached_buckets=max_cached_buckets
        self.cache_evictions=self.cache_high_water=self.chunk_size_reductions=0
        self.eviction_events=deque(maxlen=128);self.capture_events=deque(maxlen=128)
        self.minimum_observed_free_bytes=None;self.reserve_bytes=2*1024**3
        self._public_lock=threading.Lock();self._public_owner=threading.local()

    def _memory(self):
        free,total=self.torch.cuda.mem_get_info(self.device)
        self.minimum_observed_free_bytes=free if self.minimum_observed_free_bytes is None else min(self.minimum_observed_free_bytes,free)
        return free,total

    def _evict_oldest(self,reason):
        if not self.gradient_buckets:return False
        t=self.torch;t.cuda.synchronize(self.device)
        key,bucket=self.gradient_buckets.popitem(last=False)
        before=t.cuda.memory_reserved(self.device)
        graph=bucket.get('graph')
        if graph is not None:graph.reset()
        bucket.clear();del bucket;del graph
        # Empty only this process's unused allocator cache. No other process
        # is interrupted and no live graph/data allocation is reclaimed.
        t.cuda.empty_cache();after=t.cuda.memory_reserved(self.device)
        self.cache_evictions+=1
        self.eviction_events.append({'key':list(key),'reason':reason,'reserved_before':before,'reserved_after':after})
        return True

    def _new_gradient_bucket(self,size,budget,kind,initial_indices,initial_x):
        t=self.torch
        while len(self.gradient_buckets)>=self.max_cached_buckets:self._evict_oldest('LRU count bound')
        if self.device.type=='cuda':
            estimated=size*61*61*8*(256+4*budget)+64*1024**2
            free,total=self._memory()
            while estimated+self.reserve_bytes>free and self._evict_oldest('Capture estimate plus2GiB reserve'):
                free,total=self._memory()
            if estimated+self.reserve_bytes>free:
                available=max(0,free-self.reserve_bytes-64*1024**2)
                rows=available//(61*61*8*(256+4*budget))
                suggested=0 if rows<1 else 1<<(int(rows).bit_length()-1)
                raise GraphWorkspaceBudgetError('Even an empty graph cache cannot fit this shape plus2GiB reserve',suggested)
        bucket=super()._new_gradient_bucket(size,budget,kind,initial_indices,initial_x)
        self.gradient_buckets.move_to_end((size,budget,kind));self.cache_high_water=max(self.cache_high_water,len(self.gradient_buckets))
        if self.device.type=='cuda':
            free,total=self._memory()
            self.capture_events.append({'key':[size,budget,kind],'free_after':free,'total_bytes':total,
                'allocated_after':t.cuda.memory_allocated(self.device),'reserved_after':t.cuda.memory_reserved(self.device)})
            if free<self.reserve_bytes:raise MemoryError('New graph violated2GiB headroom despite conservative preflight')
        return bucket

    def _mixed_tensors(self,slots,parameters,kinds):
        while True:
            if self.device.type=='cuda':
                free,_=self._memory()
                while free<self.reserve_bytes and self._evict_oldest('Restore2GiB reserve before wave'):free,_=self._memory()
                if free<self.reserve_bytes:raise MemoryError('Resident data/external GPU use leaves less than2GiB headroom')
            try:return super()._mixed_tensors(slots,parameters,kinds)
            except GraphWorkspaceBudgetError as error:
                if error.suggested_rows<1 or error.suggested_rows>=self.batch_tasks:raise
                self.batch_tasks=error.suggested_rows;self.bucket_quantum=self.batch_tasks;self.chunk_size_reductions+=1

    def mixed_tensors(self,*args,**kwargs):
        # The inherited inner call lock still guards all Torch work. The new
        # outer guard also covers allocation/copy of shared ctypes outputs.
        if getattr(self._public_owner,'buffered',False):return super().mixed_tensors(*args,**kwargs)
        if not self._public_lock.acquire(blocking=False):raise RuntimeError('Concurrent public GPU gradient calls unsupported')
        try:return super().mixed_tensors(*args,**kwargs)
        finally:self._public_lock.release()

    def mixed_buffered(self,*args,**kwargs):
        if not self._public_lock.acquire(blocking=False):raise RuntimeError('Concurrent public GPU gradient calls unsupported')
        self._public_owner.buffered=True
        try:return super().mixed_buffered(*args,**kwargs)
        finally:self._public_owner.buffered=False;self._public_lock.release()

    def release_graph_cache(self):
        if not self._public_lock.acquire(blocking=False):raise RuntimeError('Cannot release cache during a GPU gradient call')
        try:
            while self._evict_oldest('Explicit release'):pass
        finally:self._public_lock.release()

    def graph_manifest(self):
        report=super().graph_manifest()
        report.update(schema='kakscuda.publication_gpu_gradient/2',
            shape_policy='Power of two for every chunk; arbitrary wave length is padded/scattered by verifiedv1',
            max_cached_buckets=self.max_cached_buckets,cache_high_water=self.cache_high_water,cache_evictions=self.cache_evictions,
            eviction_events_recent128=list(self.eviction_events),capture_events_recent128=list(self.capture_events),
            memory_reserve_bytes=self.reserve_bytes,minimum_observed_free_bytes=self.minimum_observed_free_bytes,
            chunk_size_reductions=self.chunk_size_reductions,effective_batch_tasks=self.batch_tasks,
            concurrent_public_call_policy='Reject across Torch evaluation and shared ctypes output copying',
            v2_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            v1_math_modified=False)
        return report
