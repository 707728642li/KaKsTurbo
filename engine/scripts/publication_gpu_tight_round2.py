"""Development-only positive-column host bound; inherited device math intact.

TightMixin works before any existing gradient provider in its MRO. Only the
native grouping call is redirected; original validation/analyze/context remain.
No Torch import, CUDA initialization, or native DLL load happens at import.
"""
from __future__ import annotations
from product_v1_platform import build_record, native_path, snapshot_directory
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 61*12+12
DLL_PATH = ROOT/'build/publication-tight-round2-v1/tight_bound.dll'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def column_coefficients(prepared):
    """Use original edge order and full unmasked mr for both pi policies."""
    def one(pi):
        if len(pi)!=61 or any(not math.isfinite(p) or p<0. for p in pi):
            raise ValueError('Expected 61 finite nonnegative frequencies')
        live=[p>0. for p in pi]
        columns=[[0.]*12 for _ in range(61)]; mr=[0.]*12
        for i,j,rate,syn in prepared.edges:
            k=rate+(0 if syn else 6)
            symmetric=math.sqrt(pi[i]*pi[j]) if live[i] and live[j] else 0.
            if live[i]:columns[i][k]+=pi[j]+symmetric
            if live[j]:columns[j][k]+=pi[i]+symmetric
            mr[k]+=(2.*pi[i])*pi[j]
        return [v for row in columns for v in row]+mr
    equal=one([1./61]*61)
    flat=[]
    for values in prepared.values:
        flat.extend(equal);flat.extend(one(values[2:63]))
    return flat

class TightContext:
    """ctypes owner with an original group ABI and a separate native context."""
    def __init__(self,prepared,dll_path=DLL_PATH):
        self.owner=threading.get_ident();self.context=C.c_void_p()
        self.path=native_path(dll_path);self.path.relative_to(ROOT)
        self.metadata=json.loads((build_record(self.path)).read_text(encoding='utf-8-sig'))
        if sha(self.path)!=self.metadata['dll_sha256']:
            raise ValueError('Tight DLL differs from its build record')
        self.before={'dll':sha(self.path),'build':sha(build_record(self.path))}
        self.source_records={}
        for name,expected in self.metadata['source_sha256'].items():
            live=ROOT/'src/gpu_round2'/name;snapshot=snapshot_directory(self.path)/name
            if sha(live)!=expected or sha(snapshot)!=expected:
                raise ValueError('Tight build source/snapshot mismatch: '+name)
            self.source_records[name]={'live':str(live),'snapshot':str(snapshot),'sha256':expected}
        self.dll=C.CDLL(str(self.path));self.error=C.create_string_buffer(512)
        self.dll.gy94_tight_host_create.argtypes=[C.c_uint32,C.c_void_p,C.POINTER(C.c_void_p),C.c_char_p,C.c_uint32]
        self.dll.gy94_tight_host_create.restype=C.c_int
        self.dll.gy94_tight_host_free.argtypes=[C.c_void_p];self.dll.gy94_tight_host_free.restype=None
        self.dll.gy94_graph_host_group.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_double),C.c_char_p,C.c_uint32]
        self.dll.gy94_graph_host_group.restype=C.c_int
        pairs=len(prepared.values)
        if not 1<=pairs<=1000:raise ValueError('Tight experiment requires 1..1000 development pairs')
        flat=column_coefficients(prepared)
        if len(flat)!=pairs*2*STRIDE:raise ValueError('Tight coefficient shape mismatch')
        self.coefficients_sha256=hashlib.sha256(bytes((C.c_double*len(flat))(*flat))).hexdigest()
        values=(C.c_double*len(flat))(*flat)
        code=self.dll.gy94_tight_host_create(pairs,values,C.byref(self.context),self.error,len(self.error))
        if code:raise ValueError(self.error.value.decode('utf-8',errors='replace'))
        self.pairs=pairs

    def group(self,ignored_old_context,*args):
        if threading.get_ident()!=self.owner or not self.context.value:
            raise RuntimeError('Tight context requires its live creating thread')
        return self.dll.gy94_graph_host_group(self.context,*args)

    def close(self):
        if self.context.value:
            if threading.get_ident()!=self.owner:raise RuntimeError('Tight close requires its creating thread')
            self.dll.gy94_tight_host_free(self.context);self.context.value=None

    def __del__(self):
        try:self.close()
        except Exception:pass

    def manifest(self):
        current={'dll':sha(self.path),'build':sha(build_record(self.path))}
        unchanged=current==self.before and all(sha(row['live'])==row['sha256']==sha(row['snapshot']) for row in self.source_records.values())
        if not unchanged:raise ValueError('Tight native source/build/DLL changed during execution')
        return {'schema':'kakscuda.tight_host_round2/1','path':str(self.path),'build':self.metadata,
            'before':self.before,'after':current,'PASS':unchanged,'source_records':self.source_records,
            'coefficient_sha256':self.coefficients_sha256,'pairs':self.pairs,'basis_shape':[self.pairs,2,61,12],
            'mr_shape':[self.pairs,2,12],'support':'All pi>0; diagonal from full Q; full unmasked mr',
            'formula':'(x0/mr)*max_column(sum_k(columns[column,k]*weights[k]))',
            'mr_order':'Original 12-term ascending left fold','safety_relative_margin':1e-12,
            'invalid_numeric_policy':'Use maximum permitted budget; original validation/device required<=budget remain authoritative',
            'device_math_modified':False,'optimizer_modified':False}

class _DLLProxy:
    def __init__(self,original,tight):self.original,self.tight=original,tight
    def __getattr__(self,name):
        if name=='gy94_graph_host_group':return self.tight.group
        return getattr(self.original,name)

class _HostProxy:
    def __init__(self,original,tight):
        self.original,self.tight=original,tight;self.dll=_DLLProxy(original.dll,tight)
    def __getattr__(self,name):return getattr(self.original,name)
    def close(self):
        try:self.tight.close()
        finally:self.original.close()

class TightMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=self.native_host
        try:
            self.tight_host=TightContext(self.prepared)
            self.native_host=_HostProxy(original,self.tight_host)
        except BaseException:
            original.close()
            if hasattr(self,'release_graph_cache'):self.release_graph_cache()
            raise
    def graph_manifest(self):
        report=super().graph_manifest()
        report['round2_tight_host']=self.tight_host.manifest()
        report['round2_tight_binding_source_sha256']=sha(__file__)
        return report
