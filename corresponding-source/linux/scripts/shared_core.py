"""Standard-library binding for the independent prepared-pair/Q shared core.

This module does not fit models, import Torch, or load a GPU runtime.
"""
from __future__ import annotations
from product_v1_platform import dll_directory, native_path
import ctypes as C
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WIDTH=11305
OFFSETS={'Q':(0,3721),'S':(3721,3721),'A':(7442,3721),'pi':(11163,64),
         'sqrt_pi':(11227,61),'rates':(11288,6),'scalars':(11294,11)}
LEGACY_CUTOFF=0
STRICT_POSITIVE=1


class SharedCore:
    def __init__(self,path=None):
        self.directories=[];self.closed=False
        try:self._initialize(path)
        except BaseException:
            self.close();raise

    def _initialize(self,path):
        self.path=native_path(Path(path or ROOT/'build/shared-core-v1-reviewed/kaks_shared_core.dll'))
        for directory in (self.path.parent,ROOT/'envs/cpu-reference/Library/bin'):
            self.directories.append(dll_directory(str(directory)))
        self.dll=C.CDLL(str(self.path));d=self.dll
        pd=C.POINTER(C.c_double);pi=C.POINTER(C.c_int32);pb=C.POINTER(C.c_int8);u=C.c_uint32
        d.kaks_core_abi_version.argtypes=[];d.kaks_core_abi_version.restype=u
        d.kaks_core_edge_count.argtypes=[];d.kaks_core_edge_count.restype=u
        if d.kaks_core_abi_version()!=1 or d.kaks_core_edge_count()!=263:raise RuntimeError('Unknown shared-core ABI/code1 edge table')
        d.kaks_core_create_prepared_pair.argtypes=[C.c_char_p,pi,pd,pb,pb,pd,u,pd,C.POINTER(C.c_void_p),C.c_char_p,u]
        d.kaks_core_export_prepared_pair.argtypes=[C.c_void_p,pi,pd,pb,pb,pd,u,pd,C.c_char_p,u]
        d.kaks_core_q_snapshot.argtypes=[C.c_void_p,u,u,pd,pd,u,C.c_char_p,u]
        d.kaks_core_free_prepared_pair.argtypes=[C.c_void_p];d.kaks_core_free_prepared_pair.restype=None
        for name in ('create_prepared_pair','export_prepared_pair','q_snapshot'):getattr(d,'kaks_core_'+name).restype=C.c_int

    @staticmethod
    def check(code,error):
        if code:raise ValueError('Shared core: '+error.value.decode('utf-8',errors='replace'))

    def prepare(self,pair_id,exported):
        if self.closed:raise RuntimeError('Shared core binding is closed')
        meta,values,z0,z1,weights,gc=exported;count=len(weights)
        meta=(C.c_int32*4)(*meta);values=(C.c_double*66)(*values);gc=(C.c_double*4)(*gc)
        z0=(C.c_int8*count)(*z0);z1=(C.c_int8*count)(*z1);weights=(C.c_double*count)(*weights)
        handle=C.c_void_p();error=C.create_string_buffer(1024)
        code=self.dll.kaks_core_create_prepared_pair(pair_id.encode('ascii'),meta,values,z0,z1,weights,count,gc,
            C.byref(handle),error,len(error));self.check(code,error)
        return PreparedPair(self,handle,count)

    def close(self):
        """After all pairs/readers finish, release search-directory cookies.

        ctypes keeps its DLL loaded; this does not manually unload code or
        free caller-owned pairs. Clearing the list makes repeated close safe.
        """
        directories,self.directories=self.directories,[];self.closed=True
        for directory in directories:directory.close()
    def __enter__(self):
        if self.closed:raise RuntimeError('Shared core binding is closed')
        return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass


class PreparedPair:
    def __init__(self,core,handle,count):self.core,self.handle,self.count=core,handle,count
    def require_open(self):
        if not self.handle.value:raise RuntimeError('Shared prepared pair is closed')
    def export(self):
        self.require_open();n=self.count
        meta=(C.c_int32*4)();values=(C.c_double*66)();gc=(C.c_double*4)()
        z0=(C.c_int8*n)();z1=(C.c_int8*n)();weights=(C.c_double*n)();error=C.create_string_buffer(1024)
        code=self.core.dll.kaks_core_export_prepared_pair(self.handle,meta,values,z0,z1,weights,n,gc,error,len(error))
        self.core.check(code,error);return meta,values,z0,z1,weights,gc
    def snapshot(self,model,x,support_policy):
        self.require_open()
        if len(x)!=7:raise ValueError('Expected seven raw parameters')
        params=(C.c_double*7)(*x);out=(C.c_double*WIDTH)();error=C.create_string_buffer(1024)
        code=self.core.dll.kaks_core_q_snapshot(self.handle,model,support_policy,params,out,WIDTH,error,len(error))
        self.core.check(code,error);return out
    def close(self):
        if self.handle.value:self.core.dll.kaks_core_free_prepared_pair(self.handle);self.handle=C.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    # Caller must join concurrent readers before explicit close/context exit.
