"""Independent FP64 CUDA Graph fused F/log-gradient provider.

Inherits the frozen publication pure Q/mr and StrictPositive frequency mapping.
Explicit T18 products have a supported autograd graph; the complete chain from
raw x through rates, full mean-rate and masked symmetric A is differentiated.
Integer scaling decisions are detached. No optimizer, clipping or FD provider.
Importing this module neither imports Torch nor starts any GPU work.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import math
from pathlib import Path
import threading
import time
from publication_gpu_likelihood import PublicationGPULikelihood
from torch_graph_likelihood import T18_COEFFICIENTS,THETA18_FP64

ROOT=Path(__file__).resolve().parents[1]

class GPUGradientError(ArithmeticError):
    def __init__(self,message,objectives=None,gradient_log=None,diagnostics=None,codes=None):
        super().__init__(message)
        self.objectives,self.gradient_log,self.diagnostics,self.codes=objectives,gradient_log,diagnostics,codes

def explicit_t18_ad(a,identity,budget):
    """Complete batched T18 and all squarings; detach integer scaling only.

    Guard failures are represented by NaN, never by an accepted truncated
    polynomial. The boundary correction mirrors the shared CPU implementation.
    Explicit coefficient folds avoid the private linear-combination backward.
    """
    import torch as t
    norm=a.detach().abs().sum(dim=-2).amax(dim=-1)
    required=t.ceil(t.log2(norm)-math.log2(THETA18_FP64)).clamp_min(0.)
    required=required+(norm*t.pow(2.,-required)>THETA18_FP64).to(t.float64)
    valid=t.isfinite(norm)&t.isfinite(required)&(required<=budget)
    selected=t.where(valid,required,t.zeros_like(required))
    safe=t.where(valid[:,None,None],a,t.zeros_like(a))
    z=safe*t.pow(2.,-selected)[:,None,None]
    a2=z@z;a3=z@a2;a6=a3@a3
    powers=(identity.expand_as(a),z,a2,a3,a6);bs=[]
    for row in T18_COEFFICIENTS:
        value=t.zeros_like(a)
        for power,coefficient in zip(powers,row):value=value+power*coefficient
        bs.append(value)
    c=bs[3]+bs[0]@bs[4];result=bs[1]+(bs[2]+c)@c
    for step in range(budget):
        square=result@result
        result=t.where((selected>step)[:,None,None],square,result)
    result=t.where(valid[:,None,None],result,t.full_like(result,float('nan')))
    return result,valid,norm,selected,required

class PublicationGPUGradient(PublicationGPULikelihood):
    """Single-call owner of resident data and lazy kind/squaring graph buckets.

    Kind0 returns raw F and zero gradient. Kind1 returns raw F and unnormalized
    g_log=x*dF/draw_x, including the full Q/mr quotient. Inactive entries zero.
    Repeated slots and mixed kinds preserve requested row order. The optimizer
    alone divides F and gradient by codon count L. Numeric failed rows retain
    code1/NaN while other rows remain valid; caller may explicitly use another
    validated provider. This version's fallback policy is explicit rejection.
    """
    def __init__(self,prepared,device='cuda:1',batch_tasks=512,max_squarings=6,
                 use_graphs=True,adaptive_squarings=True,bucket_quantum=32,**options):
        fallback=options.pop('fallback','raise')
        if fallback!='raise':raise ValueError('Fused GPU gradient supports explicit rejection fallback only')
        super().__init__(prepared,device,batch_tasks,max_squarings=max_squarings,use_graphs=use_graphs,
            fallback='raise',adaptive_squarings=adaptive_squarings,bucket_quantum=bucket_quantum,**options)
        self.gradient_buckets={};self.gradient_graph_captures=self.gradient_graph_replays=0
        self.kind_requests=[0,0];self.failed_requests=0;self._call_lock=threading.Lock()
        self.output_capacity=0;self.output_F=self.output_g=None

    def _forward_gradient(self,indices,x,budget,kind):
        t=self.torch
        a,pi,sqrt_pi,live_outer=self._matrix_inputs(indices,x)
        exp,valid,norm,actual,required=explicit_t18_ad(a,self.identity,budget)
        F=self._objective_from_exp(indices,exp,pi,sqrt_pi,live_outer)
        if kind:
            raw,=t.autograd.grad(F.sum(),x,create_graph=False,retain_graph=False)
            active=t.arange(7,device=self.device)[None,:]<(self.nkappa[indices]+2)[:,None]
            g=t.where(active,raw*x,t.zeros_like(raw))
            valid=valid&t.isfinite(g).all(dim=-1)
        else:g=t.zeros((len(indices),7),dtype=t.float64,device=self.device)
        valid=valid&t.isfinite(F)
        # Columns: raw F, valid, actual norm, required scaling, g_log7.
        return t.cat((F[:,None],valid.to(t.float64)[:,None],norm[:,None],required[:,None],g),dim=1)

    def _new_gradient_bucket(self,size,budget,kind,initial_indices,initial_x):
        t=self.torch
        if self.device.type=='cuda':
            free,_=t.cuda.mem_get_info(self.device)
            estimated=size*61*61*8*(256+4*budget)+64*1024**2
            if estimated+2*1024**3>free:raise MemoryError('Fused gradient graph estimate plus2GiB reserve exceeds available GPU memory')
        indices=initial_indices.clone();x=initial_x.detach().clone().requires_grad_(bool(kind))
        with t.inference_mode(False),t.enable_grad():
            if self.use_graphs:
                stream=t.cuda.Stream(device=self.device);stream.wait_stream(t.cuda.current_stream(self.device))
                with t.cuda.stream(stream):
                    for _ in range(3):self._forward_gradient(indices,x,budget,kind)
                t.cuda.current_stream(self.device).wait_stream(stream)
                graph=t.cuda.CUDAGraph()
                with t.cuda.graph(graph,stream=stream):result=self._forward_gradient(indices,x,budget,kind)
                bucket={'indices':indices,'x':x,'graph':graph,'result':result,'stream':stream}
                self.gradient_graph_captures+=1
            else:bucket={'indices':indices,'x':x,'graph':None}
        pinned=self.device.type=='cuda'
        bucket['host_indices']=t.empty(size,dtype=t.int64,device='cpu',pin_memory=pinned)
        bucket['host_x']=t.empty((size,7),dtype=t.float64,device='cpu',pin_memory=pinned)
        self.gradient_buckets[(size,budget,kind)]=bucket;return bucket

    def mixed_tensors(self,slots,parameters,kinds):
        """Contiguous CPU tensor requests -> F[count],g_log[count,7] tensors."""
        if not self._call_lock.acquire(blocking=False):raise RuntimeError('Concurrent calls on one GPU gradient provider unsupported')
        try:return self._mixed_tensors(slots,parameters,kinds)
        finally:self._call_lock.release()

    def _mixed_tensors(self,slots,parameters,kinds):
        t=self.torch;count=len(slots)
        if slots.device.type!='cpu' or slots.dtype!=t.int64 or slots.shape!=(count,) or not slots.is_contiguous():raise ValueError('Expected contiguous CPU int64 slots')
        if parameters.device.type!='cpu' or parameters.dtype!=t.float64 or parameters.shape!=(count,7) or not parameters.is_contiguous():raise ValueError('Expected contiguous CPU FP64 x7')
        if kinds.device.type!='cpu' or kinds.dtype!=t.int64 or kinds.shape!=(count,) or not kinds.is_contiguous():raise ValueError('Expected contiguous CPU int64 kinds')
        if not count:return t.empty(0,dtype=t.float64),t.empty((0,7),dtype=t.float64)
        if not bool(((kinds==0)|(kinds==1)).all()):raise ValueError('Expected request kinds0/1')
        self.native_host.validate(slots,parameters)
        if count>len(self.group_order):
            self.group_order=t.empty(count,dtype=t.int64,device='cpu');self.group_levels=t.empty(count,dtype=t.int32,device='cpu')
        self.native_host._check(self.native_host.dll.gy94_graph_host_group(self.native_host.context,slots.data_ptr(),parameters.data_ptr(),count,
            self.max_squarings,self.adaptive_squarings,self.group_order.data_ptr(),self.group_levels.data_ptr(),self.group_offsets,
            C.byref(self.native_host.maximum),self.native_host.error,len(self.native_host.error)))
        order=self.group_order[:count];grouped_slots=slots.index_select(0,order);grouped_x=parameters.index_select(0,order)
        grouped_kinds=kinds.index_select(0,order)
        f=t.empty(count,dtype=t.float64,device='cpu');g=t.empty((count,7),dtype=t.float64,device='cpu')
        diagnostics=t.empty((count,4),dtype=t.float64,device='cpu');codes=t.zeros(count,dtype=t.int64,device='cpu')
        with t.inference_mode(False),t.enable_grad():
            for budget in range(self.max_squarings+1):
                begin,end=self.group_offsets[budget],self.group_offsets[budget+1]
                for kind in (0,1):
                    selection=t.nonzero(grouped_kinds[begin:end]==kind).flatten()+begin
                    for offset in range(0,len(selection),self.batch_tasks):
                        selected=selection[offset:offset+self.batch_tasks]
                        local_count=len(selected);indices=grouped_slots.index_select(0,selected)
                        x=grouped_x.index_select(0,selected);local_order=order.index_select(0,selected)
                        size=(1<<(local_count-1).bit_length()) if local_count<=self.bucket_quantum else ((local_count+self.bucket_quantum-1)//self.bucket_quantum)*self.bucket_quantum
                        bucket=self.gradient_buckets.get((size,budget,kind))
                        if bucket is None:
                            incoming_indices=t.empty(size,dtype=t.int64,device=self.device);incoming_x=t.empty((size,7),dtype=t.float64,device=self.device)
                            incoming_indices[:local_count].copy_(indices);incoming_x[:local_count].copy_(x)
                            if local_count<size:
                                incoming_indices[local_count:].copy_(incoming_indices[:1].expand(size-local_count))
                                incoming_x[local_count:].copy_(incoming_x[:1].expand(size-local_count,7))
                            bucket=self._new_gradient_bucket(size,budget,kind,incoming_indices,incoming_x)
                        else:
                            bucket['host_indices'][:local_count].copy_(indices);bucket['host_x'][:local_count].copy_(x)
                            if local_count<size:
                                bucket['host_indices'][local_count:].copy_(indices[:1].expand(size-local_count))
                                bucket['host_x'][local_count:].copy_(x[:1].expand(size-local_count,7))
                            with t.no_grad():
                                bucket['indices'].copy_(bucket['host_indices'],non_blocking=True);bucket['x'].copy_(bucket['host_x'],non_blocking=True)
                        if bucket['graph'] is None:result=self._forward_gradient(bucket['indices'],bucket['x'],budget,kind)
                        else:bucket['graph'].replay();self.gradient_graph_replays+=1;result=bucket['result']
                        host=result[:local_count].detach().cpu().contiguous();bad=host[:,1]!=1.
                        # Preserve successful rows; every rejected row explicitly NaN.
                        f.index_copy_(0,local_order,t.where(bad,t.full_like(host[:,0],math.nan),host[:,0]))
                        g.index_copy_(0,local_order,t.where(bad[:,None],t.full_like(host[:,4:],math.nan),host[:,4:]))
                        diagnostics.index_copy_(0,local_order,host[:,:4]);codes.index_copy_(0,local_order,bad.to(t.int64))
                        self.kind_requests[kind]+=local_count;self.failed_requests+=int(bad.sum())
                        self.parameter_transfer_bytes+=size*64;self.objective_transfer_bytes+=local_count*11*8
        self.requests+=count;self.last_diagnostics=diagnostics;self.last_codes=codes
        if bool(codes.any()):raise GPUGradientError('Fused GPU F/gradient rows failed; inspect scaling/probability diagnostics; no clipping or truncation',
            objectives=f.tolist(),gradient_log=g.tolist(),diagnostics=diagnostics.tolist(),codes=codes.tolist())
        return f.contiguous(),g.contiguous()

    def mixed_buffered(self,slots,parameters,kinds,count=None,diagnostics=False):
        """Borrow ctypes x7/slots/kinds, return exact-size ctypes F/flat g views.

        Output views remain valid until this method is called again. Input and
        frozen prepared arrays remain live throughout the synchronous call.
        """
        t=self.torch
        if not isinstance(slots,C.Array) or slots._type_ is not C.c_int64:raise ValueError('Expected ctypes int64 slots')
        if not isinstance(parameters,C.Array) or parameters._type_ is not C.c_double:raise ValueError('Expected ctypes FP64 parameters')
        if not isinstance(kinds,C.Array) or kinds._type_ is not C.c_uint32:raise ValueError('Expected ctypes uint32 kinds')
        count=len(slots) if count is None else count
        if not isinstance(count,int) or count<0 or count>len(slots) or count>len(kinds) or count*7>len(parameters):raise ValueError('Invalid borrowed count/capacity')
        if not count:return (C.c_double*0)(),(C.c_double*0)()
        slots_t=t.frombuffer(slots,dtype=t.int64,count=count)
        x_t=t.frombuffer(parameters,dtype=t.float64,count=count*7).reshape(count,7)
        # Torch2.5 has no portable uint32 gather/index-selection implementation.
        kinds_t=t.frombuffer(kinds,dtype=t.int32,count=count).to(t.int64)
        f,g=self.mixed_tensors(slots_t,x_t,kinds_t)
        if count>self.output_capacity:
            self.output_capacity=max(count,max(64,self.output_capacity*2))
            self.output_F=(C.c_double*self.output_capacity)();self.output_g=(C.c_double*(self.output_capacity*7))()
        C.memmove(self.output_F,f.data_ptr(),count*8);C.memmove(self.output_g,g.data_ptr(),count*7*8)
        return (C.c_double*count).from_buffer(self.output_F),(C.c_double*(count*7)).from_buffer(self.output_g)

    def graph_manifest(self):
        report=super().graph_manifest()
        report.update(schema='kakscuda.publication_gpu_gradient/1',gradient_coordinate='raw F derivative w.r.t log(raw x); not divided by L; inactive zero',
            gradient_scope='Graph-compatible autograd through pure raw rates/full mean-rate and StrictPositive Q/S/A',
            integer_scaling='Detached actual norm bucket; log2 boundary correction; guarded required<=budget; never truncate',
            linear_combination='Explicit five-term coefficient fold; supported autograd ops',
            gradient_graph_captures=self.gradient_graph_captures,gradient_graph_replays=self.gradient_graph_replays,
            gradient_bucket_keys=sorted(self.gradient_buckets),kind_requests=self.kind_requests,failed_requests=self.failed_requests,
            fallback='Explicit numeric row failure; caller may use a separately validated provider',
            TF32=self.torch.backends.cuda.matmul.allow_tf32,optimizer_called=False,
            provider_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return report
