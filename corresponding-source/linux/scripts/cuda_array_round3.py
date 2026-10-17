"""Native CUDA Driver arrays/streams/graphs; explicitly NOT a Torch backend.

Only importing this stdlib module does not load CUDA. Construction selects a
physical UUID and creates a private context after an nvidia-smi snapshot.
Graph allocation capture replays a preallocated exact-sequence allocation pool.
No CUDA allocations/frees, pinned allocation or host upload occurs in capture.
All resources are explicit/process owned; no destructor invokes CUDA.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver
from array import array
from contextlib import contextmanager
import ctypes as C
from dataclasses import dataclass
import hashlib
import math
import operator
from pathlib import Path
import subprocess
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
CUDA_HEADER=ROOT/'envs/cuda-torch/include/cuda.h'
NOT_READY=600


@dataclass(frozen=True)
class NativeDType:
    name:str
    itemsize:int
    format:str
float64=NativeDType('float64',8,'d')
int64=NativeDType('int64',8,'q')
int32=NativeDType('int32',4,'i')
bool_dtype=NativeDType('bool',1,'B')
DTYPES=(float64,int64,int32,bool_dtype)


@dataclass(frozen=True)
class NativeDevice:
    type:str
    index:int|None=None
    owner_token:int|None=None
    def __str__(self):return self.type if self.index is None else f'{self.type}:{self.index}'


def shape_tuple(shape):
    if isinstance(shape,int):shape=(shape,)
    shape=tuple(operator.index(v) for v in shape)
    if any(v<0 for v in shape):raise ValueError('Allocation dimensions must be nonnegative')
    return shape
def contiguous_strides(shape):
    result=[];stride=1
    for dimension in reversed(shape):result.append(stride);stride*=max(1,dimension)
    return tuple(reversed(result))
def reshape_shape(old,requested):
    if len(requested)==1 and isinstance(requested[0],(tuple,list)):requested=tuple(requested[0])
    requested=tuple(operator.index(v) for v in requested);size=math.prod(old)
    if requested.count(-1)>1 or any(v< -1 for v in requested):raise ValueError('Invalid reshape dimensions')
    if -1 in requested:
        known=math.prod(v for v in requested if v!=-1)
        if not known or size%known:raise ValueError('Cannot infer reshape dimension')
        requested=tuple(size//known if v==-1 else v for v in requested)
    if math.prod(requested)!=size:raise ValueError('Reshape changes number of elements')
    return requested


class _UnusedAutogradFunction:
    @classmethod
    def apply(cls,*args,**kwargs):raise RuntimeError('NativeCUDAArrayAPI has no autograd; unused wrapper was invoked')


class _DeviceAllocation:
    def __init__(self,api,pointer,nbytes):
        self.api,self.pointer,self.nbytes=api,int(pointer),int(nbytes)
        self.closed=False;self.graph_users=set();self.pool=None


class GpuArray:
    def __init__(self,allocation,shape,dtype,device,strides=None,offset=0):
        self.allocation=allocation;self.shape=shape_tuple(shape);self.dtype=dtype;self.device=device
        self.strides=contiguous_strides(self.shape) if strides is None else tuple(strides)
        self.offset=operator.index(offset)
        if dtype not in DTYPES or len(self.strides)!=len(self.shape) or any(v<0 for v in self.strides):raise ValueError('Invalid native array dtype/strides')
        span=0 if self.numel()==0 else (1+sum((n-1)*s for n,s in zip(self.shape,self.strides)))*dtype.itemsize
        if self.offset<0 or self.offset+span>allocation.nbytes:raise ValueError('Array view exceeds its allocation')
    @property
    def ndim(self):return len(self.shape)
    @property
    def nbytes(self):return self.numel()*self.dtype.itemsize
    def numel(self):return math.prod(self.shape)
    def stride(self,dim=None):return self.strides if dim is None else self.strides[dim]
    def data_ptr(self):
        if self.allocation.closed:raise RuntimeError('Device allocation is closed')
        self.allocation.api._capture_reference(self.allocation)
        return self.allocation.pointer+self.offset
    def is_contiguous(self):
        expected=1
        for n,s in zip(reversed(self.shape),reversed(self.strides)):
            if n!=1 and s!=expected:return False
            expected*=max(1,n)
        return True
    def contiguous(self):
        if not self.is_contiguous():raise ValueError('Native API has no noncontiguous copy kernel')
        return self
    def transpose(self,dim0,dim1):
        dim0=operator.index(dim0);dim1=operator.index(dim1)
        if not -self.ndim<=dim0<self.ndim or not -self.ndim<=dim1<self.ndim:raise IndexError('Transpose dimension out of range')
        dim0%=self.ndim;dim1%=self.ndim
        shape=list(self.shape);strides=list(self.strides)
        shape[dim0],shape[dim1]=shape[dim1],shape[dim0];strides[dim0],strides[dim1]=strides[dim1],strides[dim0]
        return GpuArray(self.allocation,shape,self.dtype,self.device,strides,self.offset)
    def reshape(self,*shape):
        if not self.is_contiguous():raise ValueError('Native reshape requires a contiguous view')
        return GpuArray(self.allocation,reshape_shape(self.shape,shape),self.dtype,self.device,offset=self.offset)
    def unbind(self,dim=0):
        dim=operator.index(dim)
        if not -self.ndim<=dim<self.ndim:raise IndexError('Unbind dimension out of range')
        dim%=self.ndim;shape=self.shape[:dim]+self.shape[dim+1:];strides=self.strides[:dim]+self.strides[dim+1:]
        return tuple(GpuArray(self.allocation,shape,self.dtype,self.device,strides,self.offset+i*self.strides[dim]*self.dtype.itemsize) for i in range(self.shape[dim]))
    def to(self,device):
        if self.allocation.api.device(device)!=self.device:raise ValueError('Native GPU array cannot migrate devices')
        return self
    def close(self):self.allocation.api.release(self)


class HostArray:
    def __init__(self,api,memory,shape,dtype):
        self.api,self.memory,self.shape,self.dtype=api,memory,shape_tuple(shape),dtype
        self.device=NativeDevice('cpu');self.strides=contiguous_strides(self.shape)
    @property
    def ndim(self):return len(self.shape)
    def numel(self):return math.prod(self.shape)
    def stride(self,dim=None):return self.strides if dim is None else self.strides[dim]
    def reshape(self,*shape):return HostArray(self.api,self.memory,reshape_shape(self.shape,shape),self.dtype)
    def is_contiguous(self):return True
    def contiguous(self):return self
    def to(self,device):
        device=self.api.device(device)
        if device.type=='cpu':return self
        return self.api.upload(self,device)


class PinnedHostAllocation:
    def __init__(self,api,pointer,shape,dtype):
        self.api,self.pointer,self.shape,self.dtype=api,int(pointer),shape_tuple(shape),dtype
        self.nbytes=math.prod(self.shape)*dtype.itemsize;self.closed=False;self.graph_users=set()
        self._buffer=(C.c_ubyte*self.nbytes).from_address(self.pointer);self.memory=memoryview(self._buffer).cast('B')
    def data_ptr(self):
        if self.closed:raise RuntimeError('Pinned host allocation is closed')
        return self.pointer
    def close(self):
        if self.closed:return
        if self.graph_users:raise RuntimeError('Pinned host memory is retained by a live graph')
        self.api.synchronize();self.api._activate();self.api._check(self.api.lib.cuMemFreeHost(self.pointer),'cuMemFreeHost')
        self.closed=True;self.memory.release();self.api._pinned.discard(self)


class AllocationPool:
    def __init__(self,api,entries):
        self.api,self.entries=api,tuple(entries);self.closed=False
        self.nbytes=sum(allocation.nbytes for spec,allocation in self.entries)
        for spec,allocation in self.entries:allocation.pool=self
    @property
    def allocation_count(self):return len(self.entries)
    def close(self):
        if self.closed:return
        if self.api._pool_active is self:raise RuntimeError('Allocation pool is active')
        if any(allocation.graph_users for spec,allocation in self.entries):raise RuntimeError('Close graph before its allocation pool')
        self.api.synchronize()
        for spec,allocation in self.entries:self.api._free(allocation)
        self.closed=True;self.api._pools.discard(self)


class DriverStream:
    def __init__(self,api,handle):self.api,self.handle=api,handle;self.closed=False
    @property
    def cuda_stream(self):
        if self.closed:raise RuntimeError('Stream is closed')
        return self.handle.value
    def query(self):
        self.api._activate();code=self.api.lib.cuStreamQuery(self.handle)
        if code==NOT_READY:return False
        self.api._check(code,'cuStreamQuery');return True
    def synchronize(self,timeout=10.):return self.api.synchronize(timeout=timeout,stream=self)


class DriverEvent:
    def __init__(self,api,handle):self.api,self.handle,self.closed=api,handle,False
    def record(self,stream=None):
        if self.api._capturing:raise RuntimeError('Record completion/timing events outside graph capture')
        self.api._activate();stream=self.api._stream(stream);self.api._check(self.api.lib.cuEventRecord(self.handle,stream.handle),'cuEventRecord')
    def query(self):
        self.api._activate();code=self.api.lib.cuEventQuery(self.handle)
        if code==NOT_READY:return False
        self.api._check(code,'cuEventQuery');return True
    def synchronize(self,timeout=10.):
        deadline=time.monotonic()+timeout
        while not self.query():
            if time.monotonic()>=deadline:raise TimeoutError('CUDA event deadline')
            time.sleep(.001)
    def elapsed_time(self,other):
        self.api._activate();value=C.c_float();self.api._check(self.api.lib.cuEventElapsedTime(C.byref(value),self.handle,other.handle),'cuEventElapsedTime')
        return value.value
    def close(self):
        if not self.closed:
            self.api._activate();self.api._check(self.api.lib.cuEventDestroy_v2(self.handle),'cuEventDestroy_v2')
            self.closed=True;self.api._events.discard(self)


class DriverGraph:
    def __init__(self,api,graph,executable,allocations,hosts,pool):
        self.api,self.graph,self.executable=api,graph,executable;self.allocations=tuple(allocations);self.hosts=tuple(hosts);self.pool=pool
        self.closed=False;self.token=id(self)
        for allocation in self.allocations:allocation.graph_users.add(self.token)
        for host in self.hosts:host.graph_users.add(self.token)
    def launch(self,stream=None):
        if self.closed:raise RuntimeError('Graph is closed')
        self.api._activate();stream=self.api._stream(stream)
        self.api._check(self.api.lib.cuGraphLaunch(self.executable,stream.handle),'cuGraphLaunch')
    replay=launch
    def close(self):
        if self.closed:return
        self.api.synchronize();self.api._activate()
        self.api._check(self.api.lib.cuGraphExecDestroy(self.executable),'cuGraphExecDestroy')
        self.api._check(self.api.lib.cuGraphDestroy(self.graph),'cuGraphDestroy')
        for allocation in self.allocations:allocation.graph_users.discard(self.token)
        for host in self.hosts:host.graph_users.discard(self.token)
        self.closed=True;self.api._graphs.discard(self)


class _CUDAFacade:
    def __init__(self,api):self.api=api
    @contextmanager
    def device(self,device):
        self.api._validate_device(device);previous=C.c_void_p()
        self.api._check(self.api.lib.cuCtxGetCurrent(C.byref(previous)),'cuCtxGetCurrent');self.api._activate()
        try:yield
        finally:
            if previous.value!=self.api.context.value:self.api._check(self.api.lib.cuCtxSetCurrent(previous),'cuCtxSetCurrent restore')
    def current_stream(self,device=None):
        if device is not None:self.api._validate_device(device)
        self.api._activate();return getattr(self.api._local,'stream',self.api.default_stream)
    def mem_get_info(self,device=None):
        if device is not None:self.api._validate_device(device)
        return self.api.memory_info()
    @contextmanager
    def stream(self,stream):
        stream=self.api._stream(stream);previous=getattr(self.api._local,'stream',self.api.default_stream);self.api._local.stream=stream
        try:yield
        finally:self.api._local.stream=previous
    def synchronize(self,device=None):
        if device is not None:self.api._validate_device(device)
        return self.api.synchronize()


class NativeCUDAArrayAPI:
    float64=float64;int64=int64;int32=int32;bool=bool_dtype
    autograd=type('UnusedAutogradNamespace',(),{'Function':_UnusedAutogradFunction})
    def __init__(self,device='cuda:1',expected_uuid=None,reserve_bytes=2*1024**3):
        requested=str(device)
        if not requested.startswith('cuda:'):raise ValueError('Explicit logical CUDA index is required')
        logical=int(requested.split(':',1)[1])
        if logical<0 or reserve_bytes<2*1024**3:raise ValueError('Device index/reserve invalid')
        self.selected_device=NativeDevice('cuda',logical,id(self));self.reserve_bytes=reserve_bytes
        self.context=C.c_void_p();self.previous_context=C.c_void_p();self.closed=False;self._local=threading.local()
        self._allocations=set();self._pinned=set();self._streams=set();self._events=set();self._graphs=set();self._pools=set()
        self._pending_hosts=[];self._record=None;self._pool_active=None;self._pool_cursor=0;self._capturing=False
        self._capture_allocations=set();self._capture_hosts=set();self.cuda=_CUDAFacade(self)
        self.totalallocated_bytes=0;self.peakallocated_bytes=0
        snapshot=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
            text=True,capture_output=True,check=True,timeout=10.)
        self.preflight_nvidia_smi=snapshot.stdout
        gpu_rows=[tuple(part.strip() for part in line.split(',')) for line in snapshot.stdout.splitlines() if line.strip()]
        if expected_uuid is None:
            expected_uuid=next((row[1] for row in gpu_rows if int(row[0])==logical),None)
        if expected_uuid is None:raise ValueError('Requested physical GPU absent from nvidia-smi')
        wanted=uuid.UUID(str(expected_uuid).removeprefix('GPU-'));self.uuid='GPU-'+str(wanted)
        self.physical_index=next((int(row[0]) for row in gpu_rows if row[1].lower()==self.uuid.lower()),None)
        if self.physical_index is None:raise ValueError('Selected physical UUID absent from preflight')
        self.header_sha256=hashlib.sha256(CUDA_HEADER.read_bytes()).hexdigest()
        self.lib=cuda_driver();self._bind_driver()
        try:
            self._check(self.lib.cuInit(0),'cuInit');count=C.c_int();self._check(self.lib.cuDeviceGetCount(C.byref(count)),'cuDeviceGetCount')
            matched=None
            for ordinal in range(count.value):
                handle=C.c_int();raw=(C.c_ubyte*16)();self._check(self.lib.cuDeviceGet(C.byref(handle),ordinal),'cuDeviceGet')
                self._check(self.lib.cuDeviceGetUuid_v2(raw,handle),'cuDeviceGetUuid_v2')
                if bytes(raw)==wanted.bytes:matched=handle;self.driver_ordinal=ordinal;break
            if matched is None:raise RuntimeError('Selected physical UUID not found in CUDA Driver enumeration')
            name=C.create_string_buffer(256);self._check(self.lib.cuDeviceGetName(name,256,matched),'cuDeviceGetName');self.name=name.value.decode()
            version=C.c_int();self._check(self.lib.cuDriverGetVersion(C.byref(version)),'cuDriverGetVersion');self.driver_version=version.value
            self._check(self.lib.cuCtxGetCurrent(C.byref(self.previous_context)),'cuCtxGetCurrent')
            self._check(self.lib.cuCtxCreate_v2(C.byref(self.context),0,matched),'cuCtxCreate_v2')
            self.default_stream=self.create_stream()
            if self.memory_info()[0]<self.reserve_bytes:raise MemoryError('Private CUDA context lacks2GiB headroom')
        except BaseException:
            if self.context.value:
                for stream in tuple(self._streams):self.lib.cuStreamDestroy_v2(stream.handle)
                self.lib.cuCtxDestroy_v2(self.context);self.context=C.c_void_p();self.lib.cuCtxSetCurrent(self.previous_context)
            self.closed=True;raise
    def _bind_driver(self):
        P=C.c_void_p;PP=C.POINTER(P);I=C.c_int;U=C.c_uint;Q=C.c_uint64;S=C.c_size_t
        signatures={
            'cuInit':[U],'cuDeviceGetCount':[C.POINTER(I)],'cuDeviceGet':[C.POINTER(I),I],'cuDeviceGetUuid_v2':[P,I],
            'cuDeviceGetName':[P,I,I],'cuDriverGetVersion':[C.POINTER(I)],'cuGetErrorString':[I,C.POINTER(C.c_char_p)],
            'cuCtxGetCurrent':[PP],'cuCtxSetCurrent':[P],'cuCtxCreate_v2':[PP,U,I],'cuCtxDestroy_v2':[P],
            'cuStreamCreate':[PP,U],'cuStreamDestroy_v2':[P],'cuStreamQuery':[P],
            'cuMemGetInfo_v2':[C.POINTER(S),C.POINTER(S)],'cuMemAlloc_v2':[C.POINTER(Q),S],'cuMemFree_v2':[Q],
            'cuMemcpyHtoD_v2':[Q,P,S],'cuMemcpyDtoH_v2':[P,Q,S],
            'cuMemcpyHtoDAsync_v2':[Q,P,S,P],'cuMemcpyDtoHAsync_v2':[P,Q,S,P],'cuMemHostAlloc':[PP,S,U],'cuMemFreeHost':[P],
            'cuStreamBeginCapture':[P,I],'cuStreamEndCapture':[P,PP],'cuGraphInstantiateWithFlags':[PP,P,Q],
            'cuGraphLaunch':[P,P],'cuGraphExecDestroy':[P],'cuGraphDestroy':[P],
            'cuEventCreate':[PP,U],'cuEventRecord':[P,P],'cuEventQuery':[P],'cuEventDestroy_v2':[P],
            'cuEventElapsedTime':[C.POINTER(C.c_float),P,P]}
        for name,args in signatures.items():function=getattr(self.lib,name);function.argtypes=args;function.restype=I
    def _check(self,code,name):
        if code:
            message=C.c_char_p();self.lib.cuGetErrorString(code,C.byref(message))
            raise RuntimeError(f'{name} failed({code}): '+(message.value.decode() if message.value else 'unknown'))
    def _activate(self):
        if self.closed or not self.context.value:raise RuntimeError('Native CUDA context is closed')
        self._check(self.lib.cuCtxSetCurrent(self.context),'cuCtxSetCurrent')
    def device(self,value=None):
        if value is None:return self.selected_device
        if isinstance(value,NativeDevice):
            if value.type=='cuda' and value.owner_token is None and value.index==self.selected_device.index:return self.selected_device
            return value
        if str(value)=='cpu':return NativeDevice('cpu')
        if str(value)=='cuda':return self.selected_device
        if str(value).startswith('cuda:'):
            index=int(str(value).split(':',1)[1])
            return self.selected_device if index==self.selected_device.index else NativeDevice('cuda',index)
        raise ValueError('Unsupported native device')
    def _validate_device(self,device):
        if self.device(device)!=self.selected_device:raise ValueError('API owns exactly one selected logical device/physical UUID')
    def _stream(self,stream=None):
        stream=self.cuda.current_stream() if stream is None else stream
        if not isinstance(stream,DriverStream) or stream.api is not self or stream.closed:raise ValueError('Stream is not owned by this API')
        return stream
    def create_stream(self):
        if self._capturing:raise RuntimeError('No stream creation during capture')
        self._activate();handle=C.c_void_p();self._check(self.lib.cuStreamCreate(C.byref(handle),1),'cuStreamCreate')
        stream=DriverStream(self,handle);self._streams.add(stream);return stream
    def event(self,enable_timing=False):
        if self._capturing:raise RuntimeError('Create events before capture')
        self._activate();handle=C.c_void_p();self._check(self.lib.cuEventCreate(C.byref(handle),0 if enable_timing else 2),'cuEventCreate')
        event=DriverEvent(self,handle);self._events.add(event);return event
    def memory_info(self):
        if self._capturing:raise RuntimeError('No memory query during capture')
        self._activate();free,total=C.c_size_t(),C.c_size_t();self._check(self.lib.cuMemGetInfo_v2(C.byref(free),C.byref(total)),'cuMemGetInfo_v2')
        return free.value,total.value
    def _allocate(self,nbytes):
        if self._capturing:raise RuntimeError('No cuMemAlloc during capture')
        self._activate();free,total=self.memory_info()
        if nbytes>free-self.reserve_bytes:raise MemoryError('Device allocation would consume reserved2GiB')
        pointer=C.c_uint64();self._check(self.lib.cuMemAlloc_v2(C.byref(pointer),max(1,nbytes)),'cuMemAlloc_v2')
        allocation=_DeviceAllocation(self,pointer.value,max(1,nbytes));self._allocations.add(allocation)
        self.totalallocated_bytes+=allocation.nbytes;self.peakallocated_bytes=max(self.peakallocated_bytes,self.totalallocated_bytes)
        if self.memory_info()[0]<self.reserve_bytes:
            self._free(allocation);raise MemoryError('Device allocation violated2GiB reserve')
        return allocation
    def _free(self,allocation):
        if allocation.closed:return
        if self._capturing:raise RuntimeError('No cuMemFree during capture')
        if allocation.graph_users:raise RuntimeError('Allocation is retained by a live graph')
        self._activate();self._check(self.lib.cuMemFree_v2(allocation.pointer),'cuMemFree_v2')
        allocation.closed=True;self.totalallocated_bytes-=allocation.nbytes;self._allocations.discard(allocation)
    def release(self,value):
        if not isinstance(value,GpuArray) or value.allocation.api is not self:raise ValueError('Array is not API owned')
        if value.allocation.pool is not None and not value.allocation.pool.closed:raise RuntimeError('Release pool allocations through pool.close')
        if value.allocation.graph_users:raise RuntimeError('Close graph before releasing referenced inputs')
        self.synchronize();self._free(value.allocation)
    def empty(self,shape,dtype=float64,device=None):
        shape=shape_tuple(shape);device=self.device(device);self._validate_device(device)
        if dtype not in DTYPES:raise TypeError('Unsupported native dtype')
        spec=(shape,dtype,device)
        if self._pool_active is not None:
            if self._pool_cursor>=len(self._pool_active.entries):raise RuntimeError('Capture requested an extra allocation')
            expected,allocation=self._pool_active.entries[self._pool_cursor]
            if spec!=expected:raise RuntimeError(f'Capture allocation {self._pool_cursor} differs from recorded spec')
            self._pool_cursor+=1
        else:
            allocation=self._allocate(math.prod(shape)*dtype.itemsize)
            if self._record is not None:self._record.append((spec,allocation))
        return GpuArray(allocation,shape,dtype,device)
    def empty_like(self,value):return self.empty(value.shape,dtype=value.dtype,device=value.device)
    def frombuffer(self,buffer,dtype=float64,count=-1,offset=0):
        if dtype not in DTYPES:raise TypeError('Unsupported native dtype')
        memory=memoryview(buffer)
        if not memory.contiguous:raise ValueError('frombuffer requires contiguous host bytes')
        memory=memory.cast('B');offset=operator.index(offset)
        if offset<0 or offset>memory.nbytes:raise ValueError('Invalid host buffer offset')
        available=memory.nbytes-offset
        if count==-1:
            if available%dtype.itemsize:raise ValueError('Host byte length is not a multiple of dtype size')
            count=available//dtype.itemsize
        count=operator.index(count)
        if count<0 or count*dtype.itemsize>available:raise ValueError('Invalid host element count')
        return HostArray(self,memory[offset:offset+count*dtype.itemsize],(count,),dtype)
    def tensor(self,data,dtype=float64,device=None):
        def flatten(value):
            if isinstance(value,(list,tuple,array)):
                children=[flatten(v) for v in value]
                if not children:return (0,),[]
                if any(shape!=children[0][0] for shape,flat in children):raise ValueError('Ragged native tensor')
                return (len(value),*children[0][0]),[item for shape,flat in children for item in flat]
            return (),[value]
        shape,values=flatten(data)
        packed=array(dtype.format,[int(bool(v)) for v in values] if dtype==bool_dtype else values)
        return self.frombuffer(packed,dtype=dtype).reshape(shape).to(self.device(device))
    def upload(self,host,device=None):
        if self._capturing or self._pool_active is not None:raise RuntimeError('No hostupload during capture/replay allocation pass')
        device=self.device(device);self._validate_device(device);out=self.empty(host.shape,dtype=host.dtype,device=device)
        raw=bytes(host.memory);buffer=C.create_string_buffer(raw,max(1,len(raw)));self._activate()
        try:self._check(self.lib.cuMemcpyHtoD_v2(out.data_ptr(),buffer,len(raw)),'cuMemcpyHtoD_v2')
        except BaseException:self._free(out.allocation);raise
        return out
    def pinned_host(self,shape,dtype=float64):
        if self._capturing:raise RuntimeError('No pinned allocation during capture')
        shape=shape_tuple(shape);size=math.prod(shape)*dtype.itemsize
        if dtype not in DTYPES or size<=0:raise ValueError('Invalid pinned shape/dtype')
        self._activate();pointer=C.c_void_p();self._check(self.lib.cuMemHostAlloc(C.byref(pointer),size,0),'cuMemHostAlloc')
        host=PinnedHostAllocation(self,pointer.value,shape,dtype);self._pinned.add(host);return host
    def _copy_bounds(self,host,gpu,nbytes,host_offset,device_offset):
        if not isinstance(host,PinnedHostAllocation) or host.api is not self or host.closed:raise ValueError('Async copy requires API-owned pinned host memory')
        if not isinstance(gpu,GpuArray) or gpu.allocation.api is not self or not gpu.is_contiguous():raise ValueError('Async copy requires contiguous API-owned GPU array')
        nbytes=gpu.nbytes-device_offset if nbytes is None else operator.index(nbytes)
        host_offset=operator.index(host_offset);device_offset=operator.index(device_offset)
        if min(nbytes,host_offset,device_offset)<0 or host_offset+nbytes>host.nbytes or device_offset+nbytes>gpu.nbytes:raise ValueError('Async copy exceeds host/device bounds')
        return nbytes,host.pointer+host_offset,gpu.data_ptr()+device_offset
    def copy_h2d(self,dst,host,nbytes=None,host_offset=0,device_offset=0,stream=None):
        size,hptr,dptr=self._copy_bounds(host,dst,nbytes,host_offset,device_offset);self._activate();stream=self._stream(stream)
        self._check(self.lib.cuMemcpyHtoDAsync_v2(dptr,hptr,size,stream.handle),'cuMemcpyHtoDAsync_v2')
        if self._capturing:self._capture_hosts.add(host)
        else:self._pending_hosts.append((host,stream))
    def copy_d2h(self,host,src,nbytes=None,host_offset=0,device_offset=0,stream=None):
        size,hptr,dptr=self._copy_bounds(host,src,nbytes,host_offset,device_offset);self._activate();stream=self._stream(stream)
        self._check(self.lib.cuMemcpyDtoHAsync_v2(hptr,dptr,size,stream.handle),'cuMemcpyDtoHAsync_v2')
        if self._capturing:self._capture_hosts.add(host)
        else:self._pending_hosts.append((host,stream))
    def synchronize(self,timeout=10.,stream=None):
        if self._capturing:raise RuntimeError('No synchronization during capture')
        if not 0<timeout<=60:raise ValueError('Synchronization deadline must be(0,60]seconds')
        streams=tuple(self._streams) if stream is None else (self._stream(stream),);deadline=time.monotonic()+timeout
        for item in streams:
            while not item.query():
                if time.monotonic()>=deadline:raise TimeoutError('CUDA stream deadline; private context remains process owned')
                time.sleep(.001)
        self._pending_hosts=[(host,item) for host,item in self._pending_hosts if item not in streams]
    def _capture_reference(self,allocation):
        if self._capturing:self._capture_allocations.add(allocation)
    def begin_record(self):
        if self._record is not None or self._pool_active is not None or self._capturing:raise RuntimeError('Nested recording/capture forbidden')
        self._record=[]
    def end_record(self):
        if self._record is None:raise RuntimeError('No allocation recording active')
        entries=self._record;self._record=None;pool=AllocationPool(self,entries);self._pools.add(pool);return pool
    def abort_record(self):
        if self._record is not None:
            pool=self.end_record();pool.close()
    @contextmanager
    def capture_pool(self,pool):
        if pool.api is not self or pool.closed or self._pool_active is not None or self._record is not None:raise RuntimeError('Invalid/nested allocation pool')
        self._pool_active=pool;self._pool_cursor=0
        try:
            yield
            if self._capturing:raise RuntimeError('capture_end must precede allocation-pool exit')
            if self._pool_cursor!=len(pool.entries):raise RuntimeError('Capture used fewer allocations than recorded')
        except BaseException:
            if self._capturing:self.abort_capture()
            raise
        finally:self._pool_active=None;self._pool_cursor=0
    def capture_begin(self,stream=None):
        if self._capturing or self._pool_active is None:raise RuntimeError('Capture requires one active recorded pool')
        self._activate();stream=self._stream(stream);self._capture_stream=stream
        self._capture_allocations=set();self._capture_hosts=set()
        self._check(self.lib.cuStreamBeginCapture(stream.handle,0),'cuStreamBeginCapture')
        self._capture_previous_stream=getattr(self._local,'stream',self.default_stream);self._local.stream=stream;self._capturing=True
    def abort_capture(self):
        if not self._capturing:return
        self._activate()
        graph=C.c_void_p();self.lib.cuStreamEndCapture(self._capture_stream.handle,C.byref(graph));self._capturing=False
        self._local.stream=self._capture_previous_stream
        if graph.value:self.lib.cuGraphDestroy(graph)
        self._capture_allocations=set();self._capture_hosts=set()
    def capture_end(self):
        if not self._capturing:raise RuntimeError('No capture active')
        self._activate()
        if self._pool_cursor!=len(self._pool_active.entries):
            self.abort_capture();raise RuntimeError('Capture allocation sequence is incomplete')
        graph=C.c_void_p();executable=C.c_void_p();code=self.lib.cuStreamEndCapture(self._capture_stream.handle,C.byref(graph));self._capturing=False
        self._local.stream=self._capture_previous_stream
        if code:
            if graph.value:self.lib.cuGraphDestroy(graph)
            self._capture_allocations=set();self._capture_hosts=set();self._check(code,'cuStreamEndCapture')
        try:
            self._check(self.lib.cuGraphInstantiateWithFlags(C.byref(executable),graph,0),'cuGraphInstantiateWithFlags')
            if self.memory_info()[0]<self.reserve_bytes:raise MemoryError('Graph instantiation violated2GiB reserve')
        except BaseException:
            if executable.value:self.lib.cuGraphExecDestroy(executable)
            if graph.value:self.lib.cuGraphDestroy(graph)
            raise
        result=DriverGraph(self,graph,executable,self._capture_allocations,self._capture_hosts,self._pool_active)
        self._graphs.add(result);self._capture_allocations=set();self._capture_hosts=set();return result
    @property
    def allocated_bytes(self):return self.totalallocated_bytes
    @property
    def peak_allocated_bytes(self):return self.peakallocated_bytes
    def manifest(self):
        return {'schema':'kaksforge.native_CUDA_Driver_array_API_round3/1','Torch_used':False,'fake_Torch_module':False,
            'physical_GPU_UUID':self.uuid,'physical_nvidia_smi_index':self.physical_index,'CUDA_Driver_ordinal':self.driver_ordinal,
            'API_logical_device':str(self.selected_device),'device_name':self.name,'CUDA_driver_version':self.driver_version,
            'private_context':True,'nonblocking_owned_streams':len(self._streams),'reserve_bytes':self.reserve_bytes,
            'totalallocated_bytes':self.totalallocated_bytes,'peakallocated_bytes':self.peakallocated_bytes,
            'header_path':str(CUDA_HEADER),'header_sha256':self.header_sha256,'graph_allocation_policy':'Exact preallocated sequence replay; no malloc/free during capture',
            'lifetime_policy':'Graph retains every GpuArray.data_ptr allocation/pinned-copy host; close graph before pool; explicit bounded query synchronization',
            'preflight_nvidia_smi':self.preflight_nvidia_smi}
    def close(self,timeout=10.):
        if self.closed:return
        if self._capturing:self.abort_capture()
        self.synchronize(timeout=timeout)
        for graph in tuple(self._graphs):graph.close()
        for pool in tuple(self._pools):pool.close()
        for host in tuple(self._pinned):host.close()
        for event in tuple(self._events):event.close()
        for allocation in tuple(self._allocations):self._free(allocation)
        for stream in tuple(self._streams):
            self._check(self.lib.cuStreamDestroy_v2(stream.handle),'cuStreamDestroy_v2');stream.closed=True
        self._streams.clear();self._check(self.lib.cuCtxDestroy_v2(self.context),'cuCtxDestroy_v2');self.context=C.c_void_p();self.closed=True
        self._check(self.lib.cuCtxSetCurrent(self.previous_context),'cuCtxSetCurrent restore')
    def __enter__(self):return self
    def __exit__(self,*ignored):self.close()
